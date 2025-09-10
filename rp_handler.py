"""
RunPod Serverless 핸들러
"""
import runpod
from infer_core import run_infer, load_tryon_model


# Pod 부팅 시 모델 미리 로드 (콜드스타트 최소화)
print("Initializing RunPod handler...")
try:
    # fast_mode=False일 때를 위해 미리 로드
    load_tryon_model()
    print("Model preloading completed")
except Exception as e:
    print(f"Model preloading failed (will retry on demand): {e}")


def handler(job):
    """
    RunPod Serverless 핸들러 함수
    
    job["input"] 예시:
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
    print(f"Processing job: {job.get('id', 'unknown')}")
    
    # input 데이터 추출
    inputs = job.get("input", {})
    
    # 추론 실행
    result = run_infer(inputs)
    
    print(f"Job completed with status: {result.get('status', 'unknown')}")
    
    # RunPod는 결과를 그대로 반환
    return result


# RunPod Serverless 시작
if __name__ == "__main__":
    print("Starting RunPod Serverless worker...")
    runpod.serverless.start({"handler": handler})
