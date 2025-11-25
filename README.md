# ComfyUI-GoogleAIStudio

Custom nodes for integrating Google AI (Gemini) models into ComfyUI workflows.

## Installation

```bash
cd ComfyUI/custom_nodes/ComfyUI-GoogleAIStudio
pip install google-generativeai google-genai
```

Get your API key: [Google AI Studio](https://aistudio.google.com/app/apikey)

## Nodes

### 1. Google Gemini Prompt
**Category:** `Google AI`

Text generation node with vision capabilities using Gemini models.

**Features:**
- Text generation with various Gemini models (1.5, 2.0, 2.5, 3.0)
- Vision support (analyze images with text)
- System prompt customization

**Inputs:**
- `seed`: Triggers node re-execution when value changes (INT, does not guarantee reproducible results)
- `google_api_key`: Your Google AI API key (STRING)
- `llm_model`: Model selection (DROPDOWN)
- `system_prompt`: System instructions (STRING, optional)
- `user_prompt`: Main prompt (STRING)
- `image`: Image input for vision tasks (IMAGE, optional)

**Outputs:**
- `STRING`: Generated text response

---

### 2. Nanobanana Node
**Category:** `Google AI`

Image generation node using Google's Gemini image generation models.

**Features:**
- Text-to-image generation
- Image-to-image transformation
- Style transfer with reference images
- Multi-image reference support
- Model-specific image limits

**Inputs:**
- `seed`: Triggers node re-execution when value changes (INT, does not guarantee reproducible results)
- `prompt`: Image generation prompt (STRING)
- `model`: Model selection (DROPDOWN)
  - `gemini-2.0-flash-exp`: 1 image limit
  - `nanobanana`: 5 images limit
  - `nano-banana-pro-preview`: Unlimited
- `api_key`: Google AI API key (STRING)
- `images`: Reference images (IMAGE, optional)
- `temperature`: Creativity level 0.0-1.0 (FLOAT)

**Outputs:**
- `IMAGE`: Generated image(s)

---

### 3. Batch Image Normalizer
**Category:** `Google AI/Utils`

Utility node for normalizing multiple images to the same size.

**Features:**
- Dynamic input count (2-1000 images)
- Multiple resize modes
- Canvas expansion with positioning
- Aspect ratio preservation
- Resolution control

**Inputs:**
- `inputcount`: Number of image inputs (INT)
- `resize_mode`: Size determination method (DROPDOWN)
  - `largest_image`: Match largest image in batch
  - `max_resolution`: Square canvas limited to resolution_value
  - `min_resolution`: Ensure minimum resolution
  - `first_image`: Match first image size
  - `last_image`: Match last image size
- `resolution_value`: Resolution limit/target (INT)
- `upscale_method`: Interpolation method (DROPDOWN)
  - `bilinear`, `bicubic`, `nearest`, `area`, `lanczos`
- `canvas_position`: Image placement (DROPDOWN)
  - `center`, `top-left`, `top-right`, `bottom-left`, `bottom-right`
- `fill_color`: Background color (DROPDOWN)
  - `black`, `white`, `gray`, `edge_extend`
- `image_1`: First image (IMAGE, required)
- `image_2...N`: Additional images (IMAGE, optional)

**Outputs:**
- `IMAGE`: Batch of normalized images (all same size)

**Usage:**
1. Set `inputcount` to desired number
2. Click "Update inputs" button to add input slots
3. Connect images to input slots
4. Configure resize settings
5. Run to get normalized batch

---

## Example Workflows

### Text Generation
```
[Google Gemini Prompt]
├─ user_prompt: "Write a creative story"
└─ Output → [Display Text]
```

### Image Generation
```
[Nanobanana Node]
├─ prompt: "A beautiful sunset"
├─ model: "nanobanana"
└─ Output → [Preview Image]
```

### Batch Processing
```
[Load Images] → [Batch Image Normalizer]
                ├─ resize_mode: max_resolution
                ├─ resolution_value: 1024
                └─ Output → [Nanobanana Node]
```

---

## Troubleshooting

**"API key not valid"**
- Verify your API key at [Google AI Studio](https://aistudio.google.com/app/apikey)

**"google-genai package not found"**
```bash
pip install google-genai
```

**"Update inputs button not visible"**
- Restart ComfyUI to load JavaScript extensions

---

## License

Provided as-is for use with ComfyUI.

---

---

# ComfyUI-GoogleAIStudio (한국어)

ComfyUI 워크플로우에 Google AI (Gemini) 모델을 통합하는 커스텀 노드 패키지입니다.

## 설치

```bash
cd ComfyUI/custom_nodes/ComfyUI-GoogleAIStudio
pip install google-generativeai google-genai
```

API 키 발급: [Google AI Studio](https://aistudio.google.com/app/apikey)

## 노드 설명

### 1. Google Gemini Prompt
**카테고리:** `Google AI`

Gemini 모델을 사용한 비전 기능이 포함된 텍스트 생성 노드입니다.

**기능:**
- 다양한 Gemini 모델로 텍스트 생성 (1.5, 2.0, 2.5, 3.0)
- 비전 지원 (이미지를 텍스트로 분석)
- 시스템 프롬프트 커스터마이징

**입력:**
- `seed`: 값이 변경되면 노드 재실행을 트리거 (INT, 동일한 결과를 보장하지 않음)
- `google_api_key`: Google AI API 키 (STRING)
- `llm_model`: 모델 선택 (DROPDOWN)
- `system_prompt`: 시스템 지시사항 (STRING, 선택사항)
- `user_prompt`: 메인 프롬프트 (STRING)
- `image`: 비전 작업용 이미지 입력 (IMAGE, 선택사항)

**출력:**
- `STRING`: 생성된 텍스트 응답

---

### 2. Nanobanana Node
**카테고리:** `Google AI`

Google의 Gemini 이미지 생성 모델을 사용하는 이미지 생성 노드입니다.

**기능:**
- 텍스트-이미지 생성
- 이미지-이미지 변환
- 참조 이미지를 사용한 스타일 전이
- 다중 이미지 참조 지원
- 모델별 이미지 제한

**입력:**
- `seed`: 값이 변경되면 노드 재실행을 트리거 (INT, 동일한 결과를 보장하지 않음)
- `prompt`: 이미지 생성 프롬프트 (STRING)
- `model`: 모델 선택 (DROPDOWN)
  - `gemini-2.0-flash-exp`: 1장 제한
  - `nanobanana`: 5장 제한
  - `nano-banana-pro-preview`: 무제한
- `api_key`: Google AI API 키 (STRING)
- `images`: 참조 이미지 (IMAGE, 선택사항)
- `temperature`: 창의성 수준 0.0-1.0 (FLOAT)

**출력:**
- `IMAGE`: 생성된 이미지

---

### 3. Batch Image Normalizer
**카테고리:** `Google AI/Utils`

여러 이미지를 동일한 크기로 정규화하는 유틸리티 노드입니다.

**기능:**
- 동적 입력 개수 (2-1000개 이미지)
- 다양한 리사이즈 모드
- 위치 지정 가능한 캔버스 확장
- 종횡비 유지
- 해상도 제어

**입력:**
- `inputcount`: 이미지 입력 개수 (INT)
- `resize_mode`: 크기 결정 방법 (DROPDOWN)
  - `largest_image`: 배치 내 가장 큰 이미지에 맞춤
  - `max_resolution`: resolution_value로 제한된 정사각형 캔버스
  - `min_resolution`: 최소 해상도 보장
  - `first_image`: 첫 번째 이미지 크기에 맞춤
  - `last_image`: 마지막 이미지 크기에 맞춤
- `resolution_value`: 해상도 제한/목표값 (INT)
- `upscale_method`: 보간 방법 (DROPDOWN)
  - `bilinear`, `bicubic`, `nearest`, `area`, `lanczos`
- `canvas_position`: 이미지 배치 위치 (DROPDOWN)
  - `center`, `top-left`, `top-right`, `bottom-left`, `bottom-right`
- `fill_color`: 배경 색상 (DROPDOWN)
  - `black`, `white`, `gray`, `edge_extend`
- `image_1`: 첫 번째 이미지 (IMAGE, 필수)
- `image_2...N`: 추가 이미지 (IMAGE, 선택사항)

**출력:**
- `IMAGE`: 정규화된 이미지 배치 (모두 동일한 크기)

**사용 방법:**
1. `inputcount`를 원하는 개수로 설정
2. "Update inputs" 버튼을 클릭하여 입력 슬롯 추가
3. 입력 슬롯에 이미지 연결
4. 리사이즈 설정 구성
5. 실행하여 정규화된 배치 얻기

---

## 예제 워크플로우

### 텍스트 생성
```
[Google Gemini Prompt]
├─ user_prompt: "창의적인 이야기 작성"
└─ 출력 → [텍스트 표시]
```

### 이미지 생성
```
[Nanobanana Node]
├─ prompt: "아름다운 석양"
├─ model: "nanobanana"
└─ 출력 → [이미지 미리보기]
```

### 배치 처리
```
[이미지 로드] → [Batch Image Normalizer]
                ├─ resize_mode: max_resolution
                ├─ resolution_value: 1024
                └─ 출력 → [Nanobanana Node]
```

---

## 문제 해결

**"API key not valid"**
- [Google AI Studio](https://aistudio.google.com/app/apikey)에서 API 키 확인

**"google-genai package not found"**
```bash
pip install google-genai
```

**"Update inputs 버튼이 보이지 않음"**
- JavaScript 확장을 로드하기 위해 ComfyUI 재시작

---

## 라이선스

ComfyUI와 함께 사용하기 위해 제공됩니다.
