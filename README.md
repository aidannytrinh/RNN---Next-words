# RNN Dự Đoán Từ Tiếp Theo (Next-Word Prediction)

Bài tập vận dụng xây dựng một mô hình **RNN Language Model** đơn giản bằng PyTorch để dự đoán từ tiếp theo trong câu, sử dụng một đoạn văn bản tiếng Việt (không dấu) về chủ đề **nước và vòng tuần hoàn nước**.

## Mục lục

- [Tổng quan](#tổng-quan)
- [Yêu cầu môi trường](#yêu-cầu-môi-trường)
- [Cấu trúc notebook](#cấu-trúc-notebook)
- [Chi tiết kỹ thuật](#chi-tiết-kỹ-thuật)
  - [Tiền xử lý văn bản](#1-tiền-xử-lý-văn-bản)
  - [Tạo dữ liệu huấn luyện (Teacher Forcing)](#2-tạo-dữ-liệu-huấn-luyện-teacher-forcing)
  - [Kiến trúc mô hình](#3-kiến-trúc-mô-hình)
  - [Weight Tying](#4-weight-tying)
  - [Huấn luyện](#5-huấn-luyện)
  - [Suy luận / Sinh văn bản](#6-suy-luận--sinh-văn-bản)
- [Kết quả](#kết-quả)
- [Cách chạy](#cách-chạy)
- [Hạn chế và hướng mở rộng](#hạn-chế-và-hướng-mở-rộng)

## Tổng quan

Notebook `RNN_next_words.ipynb` minh họa toàn bộ pipeline của một mô hình ngôn ngữ (language model) ở mức từ (word-level), gồm các bước:

1. Tiền xử lý văn bản: xây dựng từ điển và chuyển từ thành số.
2. Xây dựng mô hình gồm `nn.Embedding`, `nn.RNN`, `nn.Linear`.
3. Áp dụng **Weight Tying** — dùng chung trọng số giữa `Embedding` và `Linear`.
4. Huấn luyện mô hình bằng `CrossEntropyLoss` và `Adam`.
5. Áp dụng **Teacher Forcing** khi tạo dữ liệu huấn luyện.
6. Dự đoán từ tiếp theo và sinh văn bản mới từ mô hình đã huấn luyện.

## Yêu cầu môi trường

- Python 3.8+
- PyTorch (`torch`)

```bash
pip install torch
```

Notebook tự động chọn `cuda` nếu có GPU khả dụng, ngược lại sẽ chạy trên `cpu`:

```python
thiet_bi = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Cấu trúc notebook

| Phần | Nội dung |
|---|---|
| 1 | Import thư viện, chọn thiết bị (`cuda`/`cpu`), đặt `torch.manual_seed(42)` |
| 2 | Định nghĩa văn bản nguồn (chủ đề vòng tuần hoàn nước) |
| 3 | Tiền xử lý văn bản → xây từ điển `tu_sang_so` / `so_sang_tu` |
| 4 | Tạo dữ liệu huấn luyện theo kiểu Teacher Forcing, đóng gói vào `DataLoader` |
| 5 | Định nghĩa mô hình `MoHinhRNN` (Embedding → RNN → Linear, có Weight Tying) |
| 6 | Vòng lặp huấn luyện với `CrossEntropyLoss` + `Adam` (300 epoch) |
| 7 | Hàm `du_doan_tu_tiep_theo`: dự đoán 1 từ kế tiếp từ một câu ngữ cảnh |
| 8 | Hàm `sinh_van_ban`: sinh liên tiếp nhiều từ (autoregressive generation) |

## Chi tiết kỹ thuật

### 1. Tiền xử lý văn bản

Hàm `tien_xu_ly_van_ban` chuyển văn bản về chữ thường, loại bỏ ký tự không phải chữ/số bằng regex (`[^a-zA-Z0-9\s]`), rồi tách thành danh sách từ (`split()`).

Từ danh sách từ, ta xây hai từ điển ánh xạ hai chiều:

- `tu_sang_so`: từ → chỉ số (index), dùng để mã hóa đầu vào cho mô hình.
- `so_sang_tu`: chỉ số → từ, dùng để giải mã đầu ra dự đoán.

Với văn bản mẫu: **127 từ**, kích thước từ điển (số từ duy nhất) là **81**.

### 2. Tạo dữ liệu huấn luyện (Teacher Forcing)

Đây là ý tưởng cốt lõi của bài toán "dự đoán từ tiếp theo": với mỗi vị trí trong chuỗi, nhãn (label) chính là chuỗi đầu vào **dịch phải 1 bước**.

```text
Đầu vào : nuoc la mot tai nguyen
Nhãn    : la mot tai nguyen quan
```

Nghĩa là tại mỗi bước thời gian, mô hình nhận **từ thật hiện tại** làm đầu vào để dự đoán **từ thật kế tiếp** — đây chính là cơ chế Teacher Forcing: trong lúc huấn luyện, mô hình luôn được "mớm" từ đúng của bước trước (ground truth) thay vì dùng từ do chính nó dự đoán, giúp quá trình học ổn định và hội tụ nhanh hơn so với việc tự hồi quy (autoregressive) ngay từ đầu.

Hàm `tao_du_lieu_huan_luyen` trượt một cửa sổ có độ dài `do_dai_ngu_canh = 5` qua toàn bộ chuỗi số để sinh ra các cặp `(dau_vao, nhan)`, sau đó đóng gói vào `TensorDataset` và `DataLoader` (`batch_size=4`, `shuffle=True`).

### 3. Kiến trúc mô hình

```python
class MoHinhRNN(nn.Module):
    def __init__(self, kich_thuoc_tu_dien, kich_thuoc_embedding, kich_thuoc_an):
        super().__init__()
        self.embedding = nn.Embedding(kich_thuoc_tu_dien, kich_thuoc_embedding)
        self.rnn = nn.RNN(kich_thuoc_embedding, kich_thuoc_an, batch_first=True)
        self.fc = nn.Linear(kich_thuoc_an, kich_thuoc_tu_dien)
        self.fc.weight = self.embedding.weight  # Weight Tying

    def forward(self, dau_vao):
        vector_tu = self.embedding(dau_vao)
        dau_ra_rnn, trang_thai_an = self.rnn(vector_tu)
        du_doan = self.fc(dau_ra_rnn)
        return du_doan
```

Mô hình gồm 3 thành phần:

1. **`nn.Embedding`** (81 → 64): chuyển mỗi chỉ số từ thành một vector dày đặc (dense vector) 64 chiều, cho phép mô hình học được các mối quan hệ ngữ nghĩa giữa các từ thay vì coi chúng là các ký hiệu rời rạc (one-hot).
2. **`nn.RNN`** (input 64, hidden 64, `batch_first=True`): xử lý tuần tự các vector embedding theo từng bước thời gian, duy trì một trạng thái ẩn (hidden state) tổng hợp thông tin ngữ cảnh từ các từ trước đó.
3. **`nn.Linear`** (64 → 81): chiếu đầu ra của RNN tại mỗi bước thời gian về không gian có số chiều bằng kích thước từ điển, tạo ra điểm số (logits) cho từng từ ứng viên.

### 4. Weight Tying

```python
self.fc.weight = self.embedding.weight
```

Ma trận trọng số của lớp `Embedding` (kích thước `[kich_thuoc_tu_dien, kich_thuoc_embedding]`) và ma trận trọng số của lớp `Linear` đầu ra (kích thước `[kich_thuoc_tu_dien, kich_thuoc_an]`) được **dùng chung** khi `kich_thuoc_embedding == kich_thuoc_an`. Ý tưởng này dựa trên trực giác rằng lớp embedding học cách biểu diễn "một từ trông như thế nào trong không gian vector", còn lớp đầu ra cần học cách "so khớp một vector ẩn với từ nào trong từ điển" — về bản chất đây là hai bài toán đối ngẫu, nên chia sẻ trọng số giúp:

- Giảm đáng kể số lượng tham số cần huấn luyện.
- Giảm nguy cơ overfitting, đặc biệt hữu ích khi dữ liệu huấn luyện nhỏ (như trong bài này).
- Cải thiện chất lượng biểu diễn từ vì trọng số được cập nhật từ cả hai luồng gradient (embedding và đầu ra).

Notebook kiểm chứng việc này bằng `mo_hinh.fc.weight is mo_hinh.embedding.weight` → `True`.

### 5. Huấn luyện

- **Hàm mất mát:** `nn.CrossEntropyLoss()` — phù hợp cho bài toán phân loại nhiều lớp (mỗi từ trong từ điển là một lớp).
- **Bộ tối ưu:** `torch.optim.Adam` với `lr=0.01`.
- **Số epoch:** 300.

Vì đầu ra của mô hình có dạng `(batch_size, do_dai_ngu_canh, kich_thuoc_tu_dien)` còn `CrossEntropyLoss` yêu cầu đầu vào 2 chiều `(N, C)` và nhãn 1 chiều `(N,)`, cả `du_doan` và `nhan` đều được `reshape` (trải phẳng) trước khi tính loss:

```python
loss = ham_mat_mat(
    du_doan.reshape(-1, kich_thuoc_tu_dien),
    nhan.reshape(-1)
)
```

### 6. Suy luận / Sinh văn bản

- `du_doan_tu_tiep_theo(mo_hinh, cau_dau_vao)`: tiền xử lý câu đầu vào, chỉ giữ lại tối đa `do_dai_ngu_canh` từ cuối cùng làm ngữ cảnh, đưa qua mô hình ở chế độ `eval()` (không tính gradient), rồi lấy `argmax` tại bước thời gian cuối cùng để chọn từ có xác suất cao nhất.
- `sinh_van_ban(mo_hinh, cau_bat_dau, so_tu_can_sinh)`: lặp lại việc dự đoán từng từ một, mỗi lần nối từ mới dự đoán được vào câu hiện tại rồi dùng câu đó làm ngữ cảnh cho bước tiếp theo (sinh văn bản kiểu tự hồi quy — autoregressive generation, khác với Teacher Forcing chỉ dùng lúc huấn luyện).

## Kết quả

Sau 300 epoch, loss trung bình giảm dần rồi dao động quanh mức thấp (do bộ dữ liệu rất nhỏ nên mô hình gần như học thuộc lòng văn bản):

```
Epoch  50 | Loss trung bình: 0.2068
Epoch 100 | Loss trung bình: 0.1868
Epoch 150 | Loss trung bình: 0.2039
Epoch 200 | Loss trung bình: 0.2070
Epoch 250 | Loss trung bình: 0.2142
Epoch 300 | Loss trung bình: 0.2196
```

Ví dụ dự đoán:

```
Câu đầu vào: nuoc la mot tai
Từ tiếp theo mô hình dự đoán: nguyen
```

Ví dụ sinh văn bản (10 từ, bắt đầu từ "nuoc la mot"):

```
nuoc la mot tai nguyen quan trong doi voi su song tren trai
```

## Cách chạy

1. Clone repo và mở notebook bằng Jupyter hoặc Google Colab:

   ```bash
   jupyter notebook RNN_next_words.ipynb
   ```

2. Chạy tuần tự từng cell từ trên xuống dưới.
3. Tùy chỉnh các siêu tham số nếu muốn thử nghiệm:
   - `do_dai_ngu_canh` (độ dài ngữ cảnh, mặc định 5)
   - `kich_thuoc_embedding`, `kich_thuoc_an` (mặc định 64)
   - `batch_size`, `lr`, `so_epoch`
4. Thử dự đoán/sinh văn bản với câu đầu vào khác bằng cách gọi `du_doan_tu_tiep_theo(...)` hoặc `sinh_van_ban(...)`.

## Hạn chế và hướng mở rộng

- **Bộ dữ liệu rất nhỏ** (127 từ, 81 từ duy nhất) nên mô hình dễ học thuộc lòng (overfit) thay vì tổng quát hóa — phù hợp cho mục đích minh họa, chưa phù hợp cho ứng dụng thực tế.
- **`nn.RNN` (vanilla RNN)** dễ gặp vấn đề vanishing/exploding gradient với chuỗi dài; có thể thay bằng `nn.LSTM` hoặc `nn.GRU` để cải thiện khả năng ghi nhớ ngữ cảnh xa.
- Chưa có tập validation/test để đánh giá khả năng tổng quát hóa của mô hình.
- Chiến lược giải mã hiện tại là **greedy decoding** (`argmax`); có thể thử `temperature sampling`, `top-k`, hoặc `beam search` để sinh văn bản đa dạng hơn.
- Có thể mở rộng bằng cách huấn luyện trên bộ dữ liệu văn bản lớn hơn, hoặc thay kiến trúc RNN bằng Transformer để so sánh hiệu năng.
