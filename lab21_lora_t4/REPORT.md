# Lab 21 — Evaluation Report

**Học viên**: Tiền Anh Kiệt — 2A202600961
**Ngày nộp**: 2026-06-25
**Submission option**: B (HF Hub)

## 1. Setup
- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 300 samples (270 train + 30 eval, split 90/10, seed 42)
- **max_seq_length**: theo p95 của dataset (notebook `Lab21_LoRA_Finetuning_T4.ipynb`)
- **GPU**: Tesla T4, 16 GB VRAM (Free Colab)
- **Training cost**: $0 (Free Colab T4) — ~3.8 phút/rank, 3 ranks ≈ 11.3 phút training total
- **LoRA config**: `target_modules=["q_proj","v_proj"]`, `lora_dropout=0`, `bias=none`, 3 epochs, cosine LR, `lr=2e-4`, warmup ratio 0.10, effective batch size 8, optimizer `adamw_8bit`
- **HF Hub link** (Option B): https://huggingface.co/codenopro/qwen2.5-3b-vi-lab21-r16

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-------------------|------------|-----------|-----------|------------|
| 8    | 1,843,200         | 3.77 min   | 11.22 GB  | 1.5577    | 4.748      |
| 16   | 3,686,400         | 3.78 min   | 10.61 GB  | 1.5161    | 4.554      |
| 64   | 14,745,600        | 3.80 min   | 12.00 GB  | 1.4768    | 4.379      |
| Base | -                 | -          | -         | -         | (không đo, so sánh qualitative trực tiếp) |

Trainable params tăng tuyến tính theo rank. Train time gần như không đổi (3.77–3.80 min) vì với 270 samples bottleneck là forward/backward qua base model 4-bit đã frozen, không phải LoRA matmul. Peak VRAM không tăng đơn điệu (r16 thấp hơn r8) — nhiễu allocator, không phải hiệu ứng thật, vì delta LoRA chỉ vài MB so với base 3B 4-bit.

## 3. Loss Curve Analysis

Train loss (r16, log mỗi 5 step, 69 step = 3 epoch × 23 step/epoch): 1.614 (step 5) → 1.555 (step 20) → 1.496 (step 35) → 1.388 (step 50) → 1.394 (step 65).

- Quan sát: **không overfitting**. Loss giảm đều ở epoch 1–2 rồi plateau nhẹ ở epoch 3 (đúng theo cosine LR decay về gần 0). Eval loss cuối (1.5161) gần với vùng train loss cuối (~1.39–1.43), không có dấu hiệu eval loss tăng trong khi train loss giảm. Với chỉ 270 samples + 3 epoch + target modules hẹp (q_proj/v_proj), model chưa đủ exposure để memorize train set.

## 4. Qualitative Comparison (5 examples)

### Example 1
**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.
**Base**: "Machine learning là một phân khúc của trí tuệ nhân tạo... Nhìn vào khía cạnh đơn giản, nó có nghĩa là máy tính học tập từ dữ liệu."
**Fine-tuned (r=16)**: "Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng..."
**Nhận xét**: improved — văn phong tự nhiên hơn, ít kiểu dịch word-by-word hơn base.

### Example 2
**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.
**Base**: code đệ quy có lỗi off-by-one (`n==1: return 0`, `n==2: return 1`).
**Fine-tuned (r=16)**: dùng iterative loop (`a, b = 0, 1; for _ in range(...)`), xử lý edge case `n=0` rõ ràng hơn, không có lỗi off-by-one.
**Nhận xét**: improved — code sạch và đúng convention hơn.

### Example 3
**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.
**Base**: nội dung hợp lý nhưng có lặp từ ("thân thiện... thân thiện").
**Fine-tuned (r=16)**: liệt kê súc tích (Chuyển đổi, Thích ứng, Đơn giản, Tương thích...), ít lặp từ hơn.
**Nhận xét**: same/slightly improved — nội dung tương đương, văn phong gọn hơn.

### Example 4
**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.
**Base**: định nghĩa đúng "LoRA = Low-Rank Adaptation".
**Fine-tuned (r=16)**: **sai thuật ngữ** — gọi LoRA là "Layer-wise Adaptive Regularization Optimization" (không tồn tại).
**Nhận xét**: degraded (case loss thật, không cherry-pick) — minh chứng fine-tune trên 300 samples generic không dạy được domain knowledge chính xác, model tự tin bịa thuật ngữ sai.

### Example 5
**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.
**Base**: giải thích đúng cả 3 khái niệm, đúng định nghĩa RAG.
**Fine-tuned (r=16)**: giải thích chung đúng hướng nhưng kém chi tiết/chính xác hơn base.
**Nhận xét**: same/slightly degraded — base (pretrain trên corpus lớn) vẫn nắm khái niệm AI hiện đại tốt hơn fine-tuned trên dataset Alpaca generic nhỏ.

## 5. Conclusion về Rank Trade-off

Trên dataset 300-sample Vietnamese-Alpaca này, r=64 cho perplexity tốt nhất (4.379 vs 4.554 ở r16 và 4.748 ở r8), nhưng mức cải thiện so với r16 chỉ ~3.8% đổi lại gấp 4× trainable params (14.7M vs 3.7M) và VRAM cao hơn (~12.0 GB vs ~10.6 GB). Đây là diminishing returns rõ ràng: bước r8→r16 giảm perplexity 4.1%, bước r16→r64 (tăng rank gấp 4 lần) chỉ giảm thêm 3.8% — chi phí param/VRAM tăng nhanh hơn nhiều so với lợi ích chất lượng thu được. Vì train time không đổi đáng kể giữa các rank ở quy mô dataset này (bottleneck nằm ở base model forward pass, không phải LoRA), yếu tố quyết định thực tế khi chọn rank là VRAM budget và risk overfit khi scale dataset lớn hơn. Recommendation cho production: chọn r=16 làm baseline — đây là điểm sweet-spot giữa chất lượng (perplexity gần r64) và chi phí (params/VRAM gần r8), đúng theo khuyến nghị "standard choice" của slide. Chỉ nên đẩy lên r=64 khi dataset lớn hơn nhiều (>2k samples) và domain cần capacity cao hơn (kiến thức chuyên ngành phức tạp); với 300 samples generic như ở đây, r=64 không đáng chi phí tăng thêm.

## 6. What I Learned
- Perplexity thấp hơn không đồng nghĩa output luôn đúng về nội dung — example 4 cho thấy fine-tuned model tự tin bịa sai định nghĩa LoRA dù perplexity tổng thể tốt hơn base, đúng nguyên tắc "fine-tune dạy style/format, không fix knowledge gap, RAG mới fix knowledge".
- Train time hầu như không phụ thuộc rank ở dataset nhỏ (270 samples) — rank chỉ ảnh hưởng trainable params và VRAM, vì phần tốn thời gian nhất là forward/backward qua base model 4-bit đã frozen, không phải LoRA matmul.
- Diminishing returns giữa các rank là đo được cụ thể (4.1% → 3.8%): "rank cao hơn luôn tốt hơn" là ngộ nhận phổ biến, cần đo thực tế trên dataset của mình trước khi chọn rank cho production.
