# Happy Mood Classifier API
## Simple FastAPI service for classifying images as "happy" or "not happy" using a pre-trained CNN.

### Deployment:
* Install requirements and run:

```
pip install -r requirements.txt
uvicorn api:app --reload
```



## Usage
Send a POST request to `/predict` with a `.npy` file (numpy array, shape 64x64x3, RGB, float32, normalized).

### Example request:
1. Prepare image:
```python
from PIL import Image
import numpy as np
img = Image.open("data/image0.png").convert("RGB").resize((64, 64))
img = np.array(img).astype(np.float32) / 255.0
np.save("test_img.npy", img)
```

2. Send request:
```
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@test_img.npy"
```

**Response:**
```
{"label": "happy", "confidence": 0.97}
```

## Endpoints
- `GET /` — API info
- `POST /predict` — image classification