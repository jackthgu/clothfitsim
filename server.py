from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import numpy as np
import cv2
import base64
import time
from typing import Optional, cast
from PIL import Image
import io
import os

from CatVTON.size_reco import pipeline as size_pipeline

app = FastAPI(title="Quick Size & Try-on")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/web", StaticFiles(directory="web", html=True), name="web")


@app.get("/")
def root():
    return RedirectResponse(url="/web/")


def to_bgr(image_bytes: bytes) -> Optional[np.ndarray]:
    if not image_bytes:
        return None
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def bgr_to_base64(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Failed to encode image")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")


def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


class TryOnEngine:
    def __init__(
        self,
        base_model_path: str = "booksforcharlie/stable-diffusion-inpainting",
        resume_repo: str = "zhengchong/CatVTON",
        width: int = 768,
        height: int = 1024,
        allow_tf32: bool = True,
    ):
        import sys  # noqa: WPS433
        pkg_dir1 = os.path.join(os.path.dirname(__file__), "CatVTON")
        pkg_dir2 = os.path.join(pkg_dir1, "CatVTON")
        for _d in (pkg_dir1, pkg_dir2):
            if os.path.isdir(_d) and _d not in sys.path:
                sys.path.insert(0, _d)

        import torch  # noqa: WPS433
        from huggingface_hub import snapshot_download  # noqa: WPS433
        from diffusers.image_processor import VaeImageProcessor  # noqa: WPS433
        from CatVTON.model.pipeline import CatVTONPipeline  # noqa: WPS433
        from CatVTON.model.cloth_masker import AutoMasker  # noqa: WPS433
        from CatVTON.utils import resize_and_crop, resize_and_padding  # noqa: WPS433

        self.torch = torch
        self.VaeImageProcessor = VaeImageProcessor
        self.CatVTONPipeline = CatVTONPipeline
        self.AutoMasker = AutoMasker
        self.resize_and_crop = resize_and_crop
        self.resize_and_padding = resize_and_padding

        self.width = width
        self.height = height

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # choose dtype
        self.weight_dtype = torch.bfloat16 if (self.device == "cuda" and torch.cuda.is_bf16_supported()) else (
            torch.float16 if (self.device == "cuda" and torch.cuda.is_available()) else torch.float32
        )

        repo_path = snapshot_download(repo_id=resume_repo)

        self.pipeline = CatVTONPipeline(
            base_ckpt=base_model_path,
            attn_ckpt=repo_path,
            attn_ckpt_version="mix",
            weight_dtype=self.weight_dtype,
            use_tf32=allow_tf32,
            device=self.device,
        )

        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True
        )
        try:
            self.automasker = AutoMasker(
                densepose_ckpt=os.path.join(repo_path, "DensePose"),
                schp_ckpt=os.path.join(repo_path, "SCHP"),
                device=self.device,
            )
        except Exception:
            self.automasker = None

    def tryon(
        self,
        person_bgr: np.ndarray,
        cloth_bgr: np.ndarray,
        cloth_type: str = "upper",
        num_inference_steps: int = 40,
        guidance_scale: float = 2.5,
        seed: Optional[int] = None,
    ) -> Image.Image:
        # Convert to PIL
        person_img = bgr_to_pil(person_bgr)
        cloth_img = bgr_to_pil(cloth_bgr)

        person_img = self.resize_and_crop(person_img, (self.width, self.height))
        cloth_img = self.resize_and_padding(cloth_img, (self.width, self.height))

        if self.automasker is not None:
            mask = self.automasker(person_img, cloth_type)["mask"]
        else:
            mask = Image.new("L", (self.width, self.height), 255)
        mask = self.mask_processor.blur(mask, blur_factor=9)

        generator = None
        if seed is not None:
            generator = self.torch.Generator(device=self.device).manual_seed(int(seed))

        result_image = self.pipeline(
            image=person_img,
            condition_image=cloth_img,
            mask=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )[0]
        return cast(Image.Image, result_image)


_tryon_engine = None


def side_by_side(p_bgr: np.ndarray, c_bgr: np.ndarray, max_h: int = 512) -> np.ndarray:
    def resize_to_h(img: np.ndarray, h: int) -> np.ndarray:
        scale = h / max(1, img.shape[0])
        w = max(1, int(img.shape[1] * scale))
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    target_h = min(max_h, max(p_bgr.shape[0], c_bgr.shape[0]))
    p = resize_to_h(p_bgr, target_h)
    c = resize_to_h(c_bgr, target_h)
    sep = np.full((target_h, 8, 3), 230, dtype=np.uint8)
    return np.hstack([p, sep, c])


@app.post("/api/predict")
async def predict(
    person_image: UploadFile = File(...),
    cloth_image: UploadFile = File(...),
    height_cm: float = Form(...),
    weight_kg: float = Form(...),
    sex: str = Form("male"),
    fit_pref: str = Form("regular"),
    item_type: str = Form("top"),
    fast_mode: bool = Form(True),
):
    t0 = time.time()

    p_bytes = await person_image.read()
    c_bytes = await cloth_image.read()
    if not p_bytes or not c_bytes:
        raise HTTPException(status_code=400, detail="Both person_image and cloth_image are required")

    p_bgr = to_bgr(p_bytes)
    c_bgr = to_bgr(c_bytes)
    if p_bgr is None or c_bgr is None:
        raise HTTPException(status_code=400, detail="Failed to decode one or more images")

    out_json, analysis = size_pipeline(p_bgr, float(height_cm), float(weight_kg), fit_pref, sex)

    # Defaults
    tryon_image_b64 = None
    tryon_error = None
    tryon_engine_name = "dummy"

    if fast_mode:
        try:
            vis_bgr = side_by_side(p_bgr, c_bgr, max_h=512)
            tryon_image_b64 = bgr_to_base64(vis_bgr)
        except Exception as e:
            tryon_error = f"quick_vis_error: {e}"
    else:
        try:
            global _tryon_engine
            if _tryon_engine is None:
                _tryon_engine = TryOnEngine(
                    base_model_path="booksforcharlie/stable-diffusion-inpainting",
                    resume_repo="zhengchong/CatVTON",
                    width=768,
                    height=1024,
                    allow_tf32=True,
                )
            cloth_type = {"top": "upper", "bottom": "lower"}.get(item_type, "overall")
            result_pil = _tryon_engine.tryon(
                p_bgr,
                c_bgr,
                cloth_type=cloth_type,
                num_inference_steps=40,
                guidance_scale=2.5,
                seed=None,
            )
            tryon_image_b64 = pil_to_base64(result_pil)
            tryon_engine_name = "CatVTON"
        except Exception as e:
            tryon_error = f"tryon_error: {e}"
            try:
                vis_bgr = side_by_side(p_bgr, c_bgr, max_h=720)
                tryon_image_b64 = bgr_to_base64(vis_bgr)
            except Exception as e2:
                tryon_error += f"; quick_vis_fallback_error: {e2}"

    return {
        "measures": {
            "chest_cm": out_json["chest_cm"],
            "waist_cm": out_json["waist_cm"],
            "hip_cm": out_json["hip_cm"],
            "shoulder_cm": out_json["shoulder_cm"],
            "inseam_cm": out_json["inseam_cm"],
            "confidence": out_json["confidence"],
        },
        "recommend": {
            "top": out_json["recommend_top"],
            "bottom": out_json["recommend_bottom"],
        },
        "analysis": analysis,
        "tryon_image_base64": tryon_image_b64,
        "tryon_engine": tryon_engine_name,
        "tryon_error": tryon_error,
        "runtime_ms": int((time.time() - t0) * 1000),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
