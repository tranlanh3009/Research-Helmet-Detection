# Research Helmet Detection

Nghiên cứu phát hiện vùng đầu **có mũ / không mũ** trong giao thông xe máy.
Đã thực hiện train baseline YOLO11n trên phần mẫu công khai EMHDD bằng CPU.
Đây là **pilot kiểm chứng pipeline**, chưa đủ dữ liệu để triển khai hoặc tự động kết luận vi phạm.

- [Scope + Dataset + Pipeline + Risk + hướng triển khai](reports/research-plan.md)
- [Audit dataset và danh sách train/validation/test](reports/dataset-audit.json)
- [Model card và kết quả thực nghiệm](reports/model-card.md)
- [Chỉ số thực tế](artifacts/metrics.json), [đánh giá ở confidence 0.25](artifacts/operating-point.json)
- [Cấu hình thí nghiệm GPU đề xuất, chưa chạy](configs/production-plan.yaml)

## Môi trường đã chuẩn bị

Checkout: `/workspace/Research-Helmet-Detection`.
Python 3.12, PyTorch 2.6.0 CPU, Ultralytics 8.3.220.
Virtualenv, dữ liệu và run logs nằm ngoài checkout ở `/workspace/helmet-runtime`.
Dataset và weights không được đưa vào Git. Checkpoint `artifacts/best.pt` và
ONNX `artifacts/best.onnx` là artifact cục bộ, được giữ trong snapshot môi trường khi publish.
Clone repository đơn thuần không bao gồm checkpoint; cần chạy lại train hoặc nhận artifact riêng.

```bash
cd /workspace/Research-Helmet-Detection
bash scripts/setup.sh
/workspace/helmet-runtime/venv/bin/python scripts/train.py \
  --data /workspace/helmet-runtime/data/emhdd-pilot/data.yaml
```

Training dùng seed 42, input 640, batch 4, tối đa 30 epoch, AdamW,
freeze 10 module đầu, early stopping patience 10. Script tạo run mới mỗi lần.
Checkpoint được chọn bằng validation; test không dùng để chọn model hay threshold.

```bash
/workspace/helmet-runtime/venv/bin/python scripts/evaluate.py \
  --data /workspace/helmet-runtime/data/emhdd-pilot/data.yaml
/workspace/helmet-runtime/venv/bin/python scripts/infer.py \
  --source /duong/dan/anh-hoac-video-cuc-bo.jpg --conf 0.25
/workspace/helmet-runtime/venv/bin/python scripts/export_onnx.py
```

Inference xuất ảnh và `detections.jsonl` vào `artifacts/predictions`.
`no_helmet` chỉ là nhãn vùng đầu; pipeline hiện chưa liên kết đầu với xe máy,
chưa tracking và chưa xác định người lái/người ngồi sau.

## Dataset và giấy phép

EMHDD: [AlanLowe007/EMHDD](https://github.com/AlanLowe007/EMHDD),
commit `137bb39cf5580c14e54e08e9ff2c7a8a420595f0`.
Bản public có **56 ảnh recognizer ban ngày**; không phải toàn bộ 52,800 ảnh tác giả mô tả.
Nhãn nguồn: `0=no_helmet`, `1=helmet`. Chưa thấy giấy phép dataset rõ ràng:
cần xin quyền trước khi phân phối dữ liệu/weights hoặc triển khai.
Ultralytics áp dụng AGPL-3.0 hoặc giấy phép Enterprise; đánh giá điều kiện phù hợp trước triển khai.
Các SHA256 đã pin bảo đảm tải lại đúng artifact đã audit, không thay thế xác minh quyền sử dụng.

MHDD có CC BY 4.0 nhưng bản đã kiểm tra chỉ có `helmet` và `motorcycle`.
Có script tải có kiểm tra SHA256 để phục vụ nghiên cứu bổ sung:

```bash
/workspace/helmet-runtime/venv/bin/python scripts/download_mhdd.py \
  --output /workspace/helmet-runtime/mhdd
```

Không suy luận “không thấy helmet” thành “người đi xe máy không đội mũ”.
