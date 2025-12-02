# Grain Analyzer Standalone

XQD 파일의 grain 분석을 수행하는 독립 실행 가능한 패키지입니다.

## 기능

- XQD 파일에서 grain 검출 및 분석
- 개별 grain 데이터 추출
- Grain 통계 계산
- 원본 및 grain_mask 포함 PDF 생성

## 설치

```bash
# 가상환경 생성 (선택사항)
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt

# 패키지 설치
pip install -e .
```

## 사용법

### Python 스크립트로 사용

```python
from pathlib import Path
from grain_analyzer.analyze import analyze_single_file_with_grain_data

xqd_file = Path("path/to/your/file.xqd")
output_dir = Path("output")

success, individual_grain_data, grain_stats, pdf_path = analyze_single_file_with_grain_data(
    xqd_file, output_dir
)

if success:
    print(f"PDF saved: {pdf_path}")
    print(f"Number of grains: {grain_stats['num_grains']}")
    print(f"First grain area: {individual_grain_data[0]['area_nm2']} nm²")
```

### 직접 import하여 사용

```python
from pathlib import Path
from grain_analyzer import analyze_single_file_with_grain_data

# 또는
from grain_analyzer.analyze import analyze_single_file_with_grain_data
```

## 출력

- **PDF 파일**: 원본 높이 데이터와 grain_mask 오버레이를 포함한 플롯
- **개별 grain 데이터**: 각 grain의 상세 정보 (면적, 직경, 중심점, peak 위치 등)
- **Grain 통계**: 전체 grain에 대한 통계 정보

## 의존성

- Python 3.8+
- numpy>=1.24.0
- matplotlib>=3.7.0
- scipy>=1.10.0
- scikit-learn>=1.3.0
- scikit-image>=0.20.0

**주의**: 이 패키지는 `xqd-analyzer`에 의존하지 않습니다. 모든 필요한 코드가 포함되어 있습니다.

## 프로젝트 구조

```
grain_analyzer/
├── grain_analyzer/
│   ├── __init__.py
│   ├── io.py               # XQD 파일 읽기
│   ├── corrections.py      # 보정 함수들
│   ├── grain_analysis.py   # Grain 분석 함수들
│   ├── utils.py            # 유틸리티 함수들
│   ├── afm_data_wrapper.py # AFMData 래퍼 클래스
│   └── analyze.py          # 메인 분석 함수
├── requirements.txt
├── setup.py
└── README.md
```

