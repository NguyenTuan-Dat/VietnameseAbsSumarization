from vncorenlp import VnCoreNLP
rdrsegmenter = VnCoreNLP("./VnCoreNLP-1.1.1.jar",
                         annotators="wseg", max_heap_size='-Xmx500m')

# Input
text = """Sau thành công ở các phim Giã Từ Dĩ Vãng, Đồng Tiền Xương Máu, Lục Vân Tiên .. và đặc biệt là vai cô Dần trong  phim điện ảnh Áo Lụa Hà Đông, Trương Ngọc Ánh dường như “quên” hẳn chốn phim trường, cô dành thời gian cho bé gái Bảo Tiên và ông xã Trần Bảo Sơn. Vai diễn mới nhất của cô trong bộ phim truyền hình Tình Yêu Và Tham Vọng dài 80 tập, hứa hẹn những bất ngờ và sự thay đổi lớn.
Với "Tình Yêu Và Tham Vọng", Trương Ngọc Ánh đã mất một tháng để đọc đi đọc lại kịch bản, trao đổi với đạo diễn, quyết định nhận vai và trở lại với màn ảnh nhỏ. Bởi nhân vật chính trong phim, ngoài cái tên Ánh như một sự trùng hợp thú vị, thì những trải nghiệm của cô ấy cũng gần như là những gì chị đã nhìn thấy, đã cảm nhận, đã trải qua. Phim được Việt hóa kịch bản từ kịch bản cùng tên của Hàn Quốc.
Cô gái Nguyễn Thị Ánh quê ở miền Trung lên thành phố làm diễn viên, nhanh chóng có được hào quang nhưng kèm theo đó là nhiều thị phi, ganh ghét. Ðây cũng là lần đầu tiên Trương Ngọc Ánh sẽ "sống" hết cuộc đời của nhân vật suốt từ khi cô ấy 18 tuổi cho đến lúc đã thành người đàn bà trên 50.
Tính cách của nhân vật lần này cũng khác, ẩn chứa bên trong vẻ ngoài mạnh mẽ, bất cần sẽ là một tâm hồn cô đơn, một trái tim mềm yếu và cần được che chở.
Phim sẽ được quay trong sáu tháng tại Phan Rang, Bình Dương và TP.HCM. Ðể chuẩn bị cho sự trở lại với "tình yêu sâu đậm" trong sáu tháng tới, hiện Trương Ngọc Ánh đã sắp xếp công việc kinh doanh, đưa mẹ vào Sài Gòn để giúp trông cháu, tranh thủ đưa cả gia đình đi chơi biển để "đền bù" trước. Có một điều chị hơi băn khoăn là nhân vật lần này nghiện rượu nặng, phải diễn thế nào cho đạt sẽ là một thử thách."""
# print(text)

# To perform word (and sentence) segmentation
sentences = rdrsegmenter.tokenize(text)
for sentence in sentences:
    print(" ".join(sentence))
