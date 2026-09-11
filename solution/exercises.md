# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature tăng dần từ 0.0 lên 1.5, các phản hồi chuyển từ trạng thái mang tính deterministic, tập trung vào các sự thật phổ biến với văn phong mẫu mực sang trạng thái phong phú, sáng tạo và đa dạng góc nhìn hơn. Ở mức 0.0 - 0.5, câu trả lời lặp lại gần như cố định và rất ổn định; ở mức 1.0, cách dùng từ sinh động và giàu cảm xúc hơn; còn ở mức 1.5, văn phong bay bổng khác lạ, bắt đầu xuất hiện câu từ liên tưởng ngẫu hứng và cấu trúc câu có phần kém chặt chẽ hơn.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt temperature trong khoảng 0.0 đến 0.3 (khuyến nghị mức 0.2). Lý do là chatbot chăm sóc và hỗ trợ khách hàng cần đặt tính chính xác, nhất quán và tuân thủ tuyệt đối quy định/chính sách sản phẩm lên hàng đầu để tránh ảo giác (hallucination); mức temperature thấp giữ phản hồi luôn chuẩn xác, an toàn nhưng vẫn giữ được sự mượt mà, tự nhiên và lịch sự trong giao tiếp.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Theo bảng giá, chi phí token output của GPT-4o ($0.010/1K token) đắt hơn xấp xỉ 16.67 lần so với GPT-4o-mini ($0.0006/1K token). Với quy mô 10.000 người dùng × 3 lượt × 350 token = 10,5 triệu token output/ngày, GPT-4o tiêu tốn khoảng $105/ngày (~$3.150/tháng), trong khi GPT-4o-mini chỉ tốn khoảng $6.30/ngày (~$189/tháng). GPT-4o xứng đáng đầu tư cho các bài toán phức tạp đòi hỏi suy luận logic đa bước sâu sắc, phân tích hợp đồng pháp lý, y tế hoặc chẩn đoán kỹ thuật quan trọng; còn GPT-4o-mini hoàn toàn lý tưởng cho các tác vụ thông thường như chatbot giải đáp câu hỏi thường gặp (FAQ), phân loại tin nhắn, gắn nhãn dữ liệu hoặc tóm tắt hội thoại ngắn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Phản hồi với persona giáo viên tiểu học có độ dài vừa phải, sử dụng từ ngữ giản dị, trong sáng và minh họa bằng hình ảnh ẩn dụ trực quan gần gũi như "cuốn sổ tay ghi chép phép màu của cả lớp mà không ai có thể tự ý tẩy xóa hay xé rách". Ngược lại, phản hồi của chuyên gia tài chính đi sâu vào chi tiết kỹ thuật với các thuật ngữ chuyên sâu như "sổ cái phân tán (distributed ledger)", "cơ chế đồng thuận (consensus mechanism)", "hàm băm mã hóa mật mã học (cryptographic hash function)" và "tính bất biến (immutability)". System prompt định hình sâu sắc hành vi mô hình bằng cách tạo một 'hệ quy chiếu' (context steering), ép mô hình điều chỉnh trường từ vựng, mức độ trừu tượng, văn phong và đối tượng tiếp nhận tương ứng.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với một đoạn văn bản tiếng Việt khoảng 100 từ, công thức ước lượng thô `số từ / 0.75` cho ra khoảng 133 token, trong khi `tiktoken` (sử dụng encoding chuẩn của GPT-4o như `o200k_base` hoặc `cl100k_base`) đếm được thực tế khoảng 165 đến 180 token, tương ứng mức chênh lệch khoảng 24% đến 35%. Tiếng Việt tốn nhiều token hơn tiếng Anh cùng độ dài vì các bộ mã hóa BPE (Byte-Pair Encoding) được xây dựng dựa trên tần suất xuất hiện chủ yếu từ các ngữ liệu tiếng Anh; hơn nữa, tiếng Việt có các dấu thanh và nguyên âm ghép đặc thù trong bảng mã UTF-8, khiến một từ tiếng Việt có dấu thường bị tách thành 2 hoặc nhiều subword tokens/bytes thay vì trọn vẹn 1 token như các từ vựng tiếng Anh thông dụng.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các ứng dụng có tương tác trực tiếp với người dùng như chatbot hội thoại thời gian thực, trợ lý ảo hoặc các tác vụ sinh nội dung văn bản dài, bởi nó giúp giảm thiểu triệt để Time to First Token (TTFT), mang lại cảm giác hệ thống phản hồi tức thì và người dùng có thể bắt đầu đọc ngay lập tức mà không phải nhìn màn hình loading trống rỗng trong nhiều giây. Ngược lại, non-streaming lại là lựa chọn phù hợp và hiệu quả hơn cho các quy trình xử lý ngầm (background jobs, batch processing), các pipeline tích hợp giữa máy chủ với máy chủ (API server-to-server), hoặc khi ứng dụng yêu cầu nhận toàn bộ đối tượng dữ liệu có cấu trúc (như JSON schema) để validate và parse trước khi tiếp tục logic nghiệp vụ.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giúp giãn cách thời gian chờ tăng theo cấp số nhân sau mỗi lần thất bại (ví dụ 0.1s -> 0.2s -> 0.4s -> 0.8s...), mang lại khoảng nghỉ tăng dần cần thiết để cụm máy chủ và cơ sở hạ tầng có cơ hội xả nghẽn và phục hồi tải. Nếu áp dụng delay cố định (như luôn chờ đúng 1 giây), hàng nghìn client cùng gặp lỗi sẽ đồng loạt gửi lại yêu cầu vào đúng thời điểm chu kỳ 1 giây tiếp theo; hiện tượng này tạo ra các đỉnh tải xung đột định kỳ (thundering herd problem), biến nỗ lực thử lại thành một đợt tấn công từ chối dịch vụ vô tình (accidental DDoS) làm hệ thống kiệt quệ kéo dài và không thể khôi phục trạng thái bình thường.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Tôi chọn persona: "Bạn là VinAI Tutor - một trợ lý học tập AI thân thiện, kiên nhẫn và chuyên nghiệp dành cho sinh viên công nghệ. Hãy luôn trả lời bằng tiếng Việt chuẩn mực, giải thích trực diện vào trọng tâm, cô đọng dưới 150 từ và đưa ra câu hỏi gợi mở để người học tự tư duy thay vì giải hộ hoàn toàn." Trong prompt này, hai lựa chọn từ ngữ quan trọng nhất là: (1) "cô đọng dưới 150 từ" giúp kiểm soát chặt chẽ độ dài token đầu ra, tiết kiệm chi phí API và tăng tốc độ phản hồi trên giao diện terminal; (2) "gợi mở để người học tự tư duy" giữ vững vai trò sư phạm của một gia sư trợ giảng, khuyến khích sinh viên tự tìm tòi kiến thức thay vì biến AI thành công cụ làm bài tập hộ.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất của trợ lý hiện tại là cơ chế cắt history thô sơ (sliding window chỉ giữ tối đa 3 lượt gần nhất = 6 messages), dẫn đến hiện tượng 'mất trí nhớ ngắn hạn' và quên mất các yêu cầu, định nghĩa quan trọng mà người dùng đã đề cập trước đó. Đề xuất cải thiện: Triển khai kỹ thuật "Tóm tắt ngữ cảnh hội thoại" (Conversation Summary Buffer) kết hợp lưu trữ bộ nhớ ngoài. Cụ thể, khi lịch sử hội thoại vượt quá ngưỡng token an toàn, hệ thống sẽ kích hoạt một lời gọi API ngầm (dùng GPT-4o-mini để tiết kiệm) tóm tắt các lượt trao đổi cũ thành một đoạn tóm lược súc tích đặt trong system message bổ sung, đồng thời vẫn duy trì nguyên vẹn 2-3 lượt chat gần nhất; cách này giúp trợ lý nhớ xuyên suốt buổi trò chuyện mà không làm bùng nổ số lượng token input.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
