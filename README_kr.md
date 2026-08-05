# Wan2.2 Remix NSFW I2V (Lightning) — RunPod Serverless

영문 문서는 [README.md](./README.md)를 참고하세요.

이 포크는 **Remix NSFW I2V v3.0 + NSFW UMT5 + Lightning 4-step** 조합의 이미지→비디오 전용 워커입니다. FLF2V(끝 프레임)는 지원하지 않습니다.

## 기본값

- `steps`: 4  
- `cfg`: 1.0  
- HIGH/LOW 분할: 2 / 2  

## 빌드

```bash
docker build -t YOUR_DOCKERHUB_USER/generate-video-nsfw-i2v:1.0 .
docker push YOUR_DOCKERHUB_USER/generate-video-nsfw-i2v:1.0
```

자세한 API·테스트 방법은 README.md를 참고하세요.
