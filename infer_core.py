"""
공통 추론 로직 모듈
FastAPI와 RunPod 핸들러 모두에서 사용
"""
import numpy as np
import cv2
import base64
import io
import os
import sys
from typing import Dict, Any, Optional, Tuple
from PIL import Image

# CatVTON 경로 추가
pkg_dir1 = os.path.join(os.path.dirname(__file__), "CatVTON")
pkg_dir2 = os.path.join(pkg_dir1, "CatVTON")
for _d in (pkg_dir1, pkg_dir2):
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)

from CatVTON.size_reco import pipeline as size_pipeline

# 전역 변수 (콜드스타트 최적화)
_tryon_engine = None


def base64_to_bgr(base64_str: str) -> Optional[np.ndarray]:
    """Base64 문자열을 BGR numpy 배열로 변환"""
    try:
        # data:image/png;base64, 제거
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        
        img_bytes = base64.b64decode(base64_str)
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"base64_to_bgr error: {e}")
        return None


def bgr_to_base64(bgr: np.ndarray) -> str:
    """BGR numpy 배열을 Base64 문자열로 변환"""
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Failed to encode image")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")


def pil_to_base64(img: Image.Image) -> str:
    """PIL Image를 Base64 문자열로 변환"""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """BGR numpy 배열을 PIL Image로 변환"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def side_by_side(p_bgr: np.ndarray, c_bgr: np.ndarray, max_h: int = 512) -> np.ndarray:
    """두 이미지를 나란히 배치 (빠른 미리보기용)"""
    def resize_to_h(img: np.ndarray, h: int) -> np.ndarray:
        scale = h / max(1, img.shape[0])
        w = max(1, int(img.shape[1] * scale))
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    target_h = min(max_h, max(p_bgr.shape[0], c_bgr.shape[0]))
    p = resize_to_h(p_bgr, target_h)
    c = resize_to_h(c_bgr, target_h)
    sep = np.full((target_h, 8, 3), 230, dtype=np.uint8)
    return np.hstack([p, sep, c])


def load_tryon_model():
    """TryOn 모델 로드 (한 번만 로드)"""
    global _tryon_engine
    
    if _tryon_engine is not None:
        return _tryon_engine
    
    print("Loading TryOn model...")
    
    import torch
    from huggingface_hub import snapshot_download
    from diffusers.image_processor import VaeImageProcessor
    from CatVTON.model.pipeline import CatVTONPipeline
    from CatVTON.model.cloth_masker import AutoMasker
    from CatVTON.utils import resize_and_crop, resize_and_padding
    
    class TryOnEngine:
        def __init__(
            self,
            base_model_path: str = "booksforcharlie/stable-diffusion-inpainting",
            resume_repo: str = "zhengchong/CatVTON",
            width: int = 768,
            height: int = 1024,
            allow_tf32: bool = True,
        ):
            self.torch = torch
            self.resize_and_crop = resize_and_crop
            self.resize_and_padding = resize_and_padding
            
            self.width = width
            self.height = height
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
            return result_image
    
    _tryon_engine = TryOnEngine()
    print("TryOn model loaded successfully")
    return _tryon_engine


def run_infer(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    메인 추론 함수
    
    payload 예시:
    {
        "person_image_base64": "...",
        "cloth_image_base64": "...",
        "height_cm": 175,
        "weight_kg": 70,
        "sex": "male",
        "fit_pref": "regular",
        "item_type": "top",
        "fast_mode": true
    }
    """
    try:
        # 입력 파싱
        person_b64 = payload.get("person_image_base64", "")
        cloth_b64 = payload.get("cloth_image_base64", "")
        height_cm = float(payload.get("height_cm", 175))
        weight_kg = float(payload.get("weight_kg", 70))
        sex = payload.get("sex", "male")
        fit_pref = payload.get("fit_pref", "regular")
        item_type = payload.get("item_type", "top")
        fast_mode = payload.get("fast_mode", True)
        
        # Base64를 BGR로 변환
        p_bgr = base64_to_bgr(person_b64)
        c_bgr = base64_to_bgr(cloth_b64)
        
        if p_bgr is None or c_bgr is None:
            return {
                "status": "error",
                "error": "Failed to decode images"
            }
        
        # 사이즈 추천 실행
        out_json, analysis = size_pipeline(p_bgr, height_cm, weight_kg, fit_pref, sex)
        
        # TryOn 실행
        tryon_image_b64 = None
        tryon_error = None
        tryon_engine_name = "dummy"
        
        if fast_mode:
            # 빠른 모드: 단순 나란히 배치
            try:
                vis_bgr = side_by_side(p_bgr, c_bgr, max_h=512)
                tryon_image_b64 = bgr_to_base64(vis_bgr)
                tryon_engine_name = "fast_preview"
            except Exception as e:
                tryon_error = f"quick_vis_error: {e}"
        else:
            # 실제 TryOn 모델 실행
            try:
                engine = load_tryon_model()
                cloth_type = {"top": "upper", "bottom": "lower"}.get(item_type, "overall")
                
                result_pil = engine.tryon(
                    p_bgr,
                    c_bgr,
                    cloth_type=cloth_type,
                    num_inference_steps=40,
                    guidance_scale=2.5,
                    seed=None,
                )
                # 결과가 Image 타입인지 확인
                if hasattr(result_pil, 'save'):  # PIL Image check
                    tryon_image_b64 = pil_to_base64(result_pil)
                else:
                    tryon_image_b64 = pil_to_base64(Image.fromarray(result_pil))
                tryon_engine_name = "CatVTON"
            except Exception as e:
                tryon_error = f"tryon_error: {e}"
                # 폴백: 나란히 배치
                try:
                    vis_bgr = side_by_side(p_bgr, c_bgr, max_h=720)
                    tryon_image_b64 = bgr_to_base64(vis_bgr)
                except Exception as e2:
                    tryon_error += f"; fallback_error: {e2}"
        
        # 결과 반환
        return {
            "status": "success",
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
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
