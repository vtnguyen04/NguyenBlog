---
title: "Giải Mã Bộ Lọc Kalman: Hành Trình Ước Lượng Trạng Thái Tối Ưu"
published: 2025-06-08
tags: ["thuật toán lọc nhiễu", "ước lượng trạng thái"]
category: "Thuật Toán lọc nhiễu"
description: "Khám phá toàn diện về bộ lọc Kalman, từ những khái niệm toán học nền tảng như xác suất, phân phối Gaussian, đến việc xây dựng mô hình, chứng minh và triển khai thuật toán đệ quy, cùng các ứng dụng thực tế trong việc lọc nhiễu và theo dõi đối tượng."
draft: false
lang: "vi"
---

## Mục lục

1. [Giới Thiệu và Đặt Vấn Đề](#phần-i-giới-thiệu-và-đặt-vấn-đề)
2. [Xây Dựng Mô Hình Toán Học](#phần-ii-xây-dựng-mô-hình-toán-học)
3. [Thuật Toán Bộ Lọc Kalman](#phần-iii-xây-dựng-và-chứng-minh-thuật-toán-bộ-lọc-kalman)
4. [Ví Dụ Minh Họa](#phần-iv-ví-dụ-minh-họa---theo-dõi-chuyển-động-1d)
5. [Kết Luận](#phần-v-kết-luận)

## Phần I: Giới Thiệu và Đặt Vấn Đề

### 1. Mở Đầu

Bộ lọc <span style="color: #e74c3c; font-weight: bold;">Kalman (Kalman Filter - KF)</span>, được đặt theo tên của Rudolf E. Kálmán, là một thuật toán đệ quy mạnh mẽ được sử dụng để ước lượng trạng thái nội tại của một hệ thống động tuyến tính từ một loạt các phép đo không hoàn hảo và chứa nhiễu. Nói một cách đơn giản, nó là một công cụ **tối ưu thống kê** để kết hợp thông tin từ nhiều nguồn không chắc chắn (mô hình dự đoán và dữ liệu đo lường) nhằm đưa ra một ước lượng tốt nhất về những gì đang thực sự xảy ra với hệ thống.

Trong bài viết này, chúng ta sẽ cùng nhau "giải phẫu" bộ lọc Kalman. Chúng ta sẽ bắt đầu từ những động lực thúc đẩy sự ra đời của nó, ôn lại các kiến thức toán học nền tảng, sau đó từng bước xây dựng nên thuật toán từ những nguyên lý cơ bản, chứng minh các công thức chính, và cuối cùng là xem xét các ví dụ minh họa cũng như ứng dụng thực tiễn của nó.

### 2. Động Lực và Nguồn Gốc Lịch Sử

Nhu cầu ước lượng và dự đoán trạng thái của các hệ thống trong điều kiện không chắc chắn đã tồn tại từ lâu trong nhiều lĩnh vực, từ thiên văn học, điều khiển học đến kinh tế. Trước bộ lọc Kalman, các phương pháp như **bình phương tối thiểu (Least Squares)** của Gauss đã được sử dụng. Tuy nhiên, phương pháp bình phương tối thiểu cổ điển thường là xử lý theo lô (batch processing), yêu cầu toàn bộ tập dữ liệu phải có sẵn để xử lý cùng một lúc. Điều này không phù hợp cho các hệ thống thay đổi theo thời gian (hệ thống động) hoặc khi dữ liệu đến tuần tự.

Bộ lọc Kalman, được giới thiệu bởi Rudolf Kálmán vào năm 1960, đã mang đến một cuộc cách mạng. Nó nổi bật với:
*   **Tính đệ quy (Recursive):** Chỉ cần ước lượng và phép đo ở bước trước đó để tính toán ước lượng hiện tại, không cần lưu trữ toàn bộ lịch sử dữ liệu.
*   **Tính tối ưu (Optimality):** Đối với các hệ thống tuyến tính có nhiễu tuân theo phân phối Gaussian, bộ lọc Kalman cung cấp ước lượng không chệch có phương sai nhỏ nhất (Minimum Mean Square Error - MMSE estimator).
*   **Xử lý thời gian thực:** Phù hợp cho các ứng dụng cần cập nhật ước lượng liên tục.

Ứng dụng nổi tiếng và tiên phong đầu tiên của bộ lọc Kalman là trong chương trình Apollo của NASA, nơi nó được sử dụng để định vị và dẫn đường cho tàu vũ trụ trên hành trình đến Mặt Trăng.

### 3. Kiến Thức Nền Tảng Cần Thiết

Để hiểu sâu sắc và xây dựng được bộ lọc Kalman, bạn cần nắm vững một số khái niệm toán học:

*   **Xác suất và Thống kê:**
    *   **Biến ngẫu nhiên và Phân phối xác suất:** Khái niệm về sự không chắc chắn và cách mô tả nó.
    *   **Phân phối Gaussian (Normal Distribution):** Đây là "trái tim" của bộ lọc Kalman chuẩn (linear Kalman filter).
        *   Một biến ngẫu nhiên vô hướng $X$ tuân theo phân phối Gaussian với trung bình (kỳ vọng) $\mu$ và phương sai $\sigma^2$ được ký hiệu là $X \sim \mathcal{N}(\mu, \sigma^2)$. Hàm mật độ xác suất (PDF) của nó là:

$$
f(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

*   Đối với vector ngẫu nhiên đa biến $\mathbf{x} \in \mathbb{R}^n$, nếu nó tuân theo phân phối Gaussian đa biến với vector kỳ vọng $\boldsymbol{\mu}$ và ma trận hiệp phương sai $\mathbf{P}$, ta ký hiệu $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \mathbf{P})$. Hàm PDF có dạng:

$$
f(\mathbf{x}; \boldsymbol{\mu}, \mathbf{P}) = \frac{1}{\sqrt{(2\pi)^n \det(\mathbf{P})}} e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \mathbf{P}^{-1}(\mathbf{x}-\boldsymbol{\mu})}
$$

* **Kỳ vọng (Mean / Expected Value):**  
  $E[X]$ là giá trị trung bình, "tâm" của phân phối.

* **Phương sai (Variance):**  
  $Var(X) = E[(X-E[X])^2]$ đo lường mức độ phân tán của biến ngẫu nhiên quanh kỳ vọng của nó.

* **Ma trận Hiệp phương sai (Covariance Matrix):**  
  Đối với vector ngẫu nhiên đa biến $\mathbf{x}$, ma trận hiệp phương sai  
  $\mathbf{P} = E[(\mathbf{x}-E[\mathbf{x}])(\mathbf{x}-E[\mathbf{x}])^T]$  
  có các phần tử trên đường chéo chính là phương sai của từng thành phần trong $\mathbf{x}$,  
  và các phần tử ngoài đường chéo chính là hiệp phương sai giữa các cặp thành phần, thể hiện mức độ tương quan tuyến tính giữa chúng.

* **Tính chất của kỳ vọng và phương sai:**
    - $E[A\mathbf{x} + B\mathbf{y} + \mathbf{c}] = A E[\mathbf{x}] + B E[\mathbf{y}] + \mathbf{c}$
    - $Var(A\mathbf{x} + \mathbf{c}) = A Var(\mathbf{x}) A^T$
    - Nếu $\mathbf{x}$ và $\mathbf{y}$ độc lập: $Var(\mathbf{x} + \mathbf{y}) = Var(\mathbf{x}) + Var(\mathbf{y})$

* **Đại số tuyến tính:**
    - Phép toán ma trận: cộng, trừ, nhân, chuyển vị (transpose), nghịch đảo (inverse).
    - Vector, không gian vector, cơ sở.
    - Định thức (determinant) của ma trận.

* **Hệ thống động (Dynamical Systems):**
    - Khái niệm về trạng thái của một hệ thống.
    - Mô hình không gian trạng thái (state-space representation) để mô tả sự tiến triển của trạng thái theo thời gian.

---

## Phần II: Xây Dựng Mô Hình Toán Học

Bộ lọc Kalman hoạt động dựa trên việc mô tả hệ thống bằng hai phương trình chính, biểu diễn dưới dạng không gian trạng thái. Chúng ta giả định rằng cả nhiễu quá trình và nhiễu đo lường đều là nhiễu trắng Gaussian, không tương quan với nhau và không tương quan với trạng thái.

### 1. Mô Tả Hệ Thống Động Tuyến Tính Rời Rạc

#### a. Mô hình Quá trình (Process Model hay State Transition Model)

Phương trình này mô tả cách trạng thái của hệ thống $\mathbf{x}_k \in \mathbb{R}^n$ tại thời điểm (rời rạc) $k$ tiến triển từ trạng thái $\mathbf{x}_{k\!-\!1}$ ở thời điểm trước đó $k-1$:

$$
\mathbf{x}_k = \mathbf{A}_k \mathbf{x}_{k-1} + \mathbf{B}_k \mathbf{u}_k + \mathbf{w}_k
$$


Trong đó:
*   $\mathbf{x}_k$: Vector trạng thái tại thời điểm $k$. Ví dụ: vị trí, vận tốc, gia tốc của một đối tượng.
*   $\mathbf{A}_k$: Ma trận chuyển đổi trạng thái (state transition matrix) từ $k-1$ sang $k$. Nó mô tả động lực học nội tại của hệ thống.
*   $\mathbf{u}_k$: Vector đầu vào điều khiển (control input vector) tại thời điểm $k$ (nếu có). Ví dụ: lực tác động lên một robot.
*   $\mathbf{B}_k$: Ma trận điều khiển đầu vào (control input matrix). Nó liên kết đầu vào điều khiển với trạng thái.
*   $\mathbf{w}_{k-1}$: Vector nhiễu quá trình (process noise vector) tại $k-1$. Đây là một vector ngẫu nhiên Gauss đa biến, giả định là nhiễu trắng với kỳ vọng bằng không và ma trận hiệp phương sai $\mathbf{Q}_k$:
    $$ \mathbf{w}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_k) $$
    Nhiễu quá trình $\mathbf{w}$ thể hiện sự không chắc chắn trong mô hình động của chúng ta, hoặc các yếu tố ngoại cảnh không được mô hình hóa. Ví dụ, nếu theo dõi một chiếc xe, $\mathbf{w}$ có thể đại diện cho những thay đổi gia tốc không lường trước do gió, mặt đường không bằng phẳng, hoặc sai số trong mô hình động lực học của xe.

#### b. Mô hình Đo lường (Measurement Model)

Phương trình này mô tả mối quan hệ giữa trạng thái thực $\mathbf{x}_k$ và phép đo $\mathbf{z}_k \in \mathbb{R}^m$ mà chúng ta thu được từ các cảm biến tại thời điểm $k$:

$$ \mathbf{z}_k = \mathbf{H}_k \mathbf{x}_k + \mathbf{v}_k $$

Trong đó:
*   $\mathbf{z}_k$: Vector đo lường tại thời điểm $k$. Ví dụ: vị trí đo được từ GPS, góc đo từ radar.
*   $\mathbf{H}_k$: Ma trận đo lường (measurement matrix hoặc observation matrix). Nó liên kết trạng thái thực với không gian đo lường.
*   $\mathbf{v}_k$: Vector nhiễu đo lường (measurement noise vector) tại $k$. Đây cũng là một vector ngẫu nhiên Gauss đa biến, giả định là nhiễu trắng với kỳ vọng bằng không và ma trận hiệp phương sai $\mathbf{R}_k$:
    $$ \mathbf{v}_k \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_k) $$
    Nhiễu đo lường $\mathbf{v}$ thể hiện sự không chính xác và nhiễu cố hữu của các cảm biến.

**Các giả định quan trọng:**
*   Nhiễu quá trình $\mathbf{w}_k$ và nhiễu đo lường $\mathbf{v}_k$ là độc lập thống kê với nhau.
*   Chúng cũng độc lập với trạng thái ban đầu $\mathbf{x}_0$.
*   Chúng là các chuỗi nhiễu trắng (uncorrelated in time).

### 2. Mục Tiêu Ước Lượng

Mục tiêu của chúng ta là tìm ra ước lượng "tốt nhất" cho trạng thái **thực ẩn** $\mathbf{x}_k$ dựa trên tất cả các thông tin có sẵn cho đến thời điểm $k$. Ký hiệu:
*   $\hat{\mathbf{x}}_{k|k-1}$: Ước lượng trạng thái **tiên nghiệm (a priori state estimate)** tại thời điểm $k$, dựa trên các phép đo đến thời điểm $k-1$. Đây là dự đoán của chúng ta về trạng thái tại $k$ *trước khi* có phép đo mới $\mathbf{z}_k$.
*   $\hat{\mathbf{x}}_{k|k}$: Ước lượng trạng thái **hậu nghiệm (a posteriori state estimate)** tại thời điểm $k$, dựa trên các phép đo đến thời điểm $k$ (bao gồm cả $\mathbf{z}_k$). Đây là ước lượng được hiệu chỉnh sau khi đã kết hợp thông tin từ phép đo mới.

"Tốt nhất" ở đây thường được hiểu theo nghĩa là tối thiểu hóa phương sai của sai số ước lượng (Minimum Mean Square Error - MMSE). Trong trường hợp nhiễu Gaussian, ước lượng MMSE cũng chính là kỳ vọng có điều kiện của trạng thái.

Cùng với ước lượng trạng thái, chúng ta cũng cần ước lượng độ không chắc chắn của các ước lượng đó, được biểu diễn bởi các ma trận hiệp phương sai sai số:

*   $\mathbf{P}_{k|k-1} = E[(\mathbf{x}_k - \hat{\mathbf{x}}_{k|k-1})(\mathbf{x}_k - \hat{\mathbf{x}}_{k|k-1})^T]$: Ma trận hiệp phương sai sai số tiên nghiệm.
*   $\mathbf{P}_{k|k} = E[(\mathbf{x}_k - \hat{\mathbf{x}}_{k|k})(\mathbf{x}_k - \hat{\mathbf{x}}_{k|k})^T]$: Ma trận hiệp phương sai sai số hậu nghiệm.

Các phần tử trên đường chéo của $\mathbf{P}$ cho biết phương sai (độ không chắc chắn) của từng thành phần trạng thái, và các phần tử ngoài đường chéo cho biết hiệp phương sai (mức độ tương quan sai số) giữa các cặp thành phần. Mục tiêu là làm cho các phần tử của $\mathbf{P}_{k|k}$ càng nhỏ càng tốt.

---

## Phần III: Xây Dựng và Chứng Minh Thuật Toán Bộ Lọc Kalman

Bộ lọc Kalman hoạt động theo một chu trình đệ quy gồm hai bước chính: **Dự đoán (Prediction)**, còn gọi là **Cập nhật Thời gian (Time Update)**, và **Hiệu chỉnh (Correction)**, còn gọi là **Cập nhật Đo lường (Measurement Update)**.

<figure class="flex flex-col items-center my-6">
  <img src="/images/kalman-process.jpg" alt="Vòng lặp bộ lọc Kalman" class="w-full md:w-3/4 lg:w-2/3 bg-white p-4 rounded-lg shadow-lg">
  <figcaption class="text-sm text-center text-neutral-600 mt-2">Hình 2: Chu trình Dự đoán - Hiệu chỉnh của Bộ lọc Kalman.</figcaption>
</figure>

### 1. Bước Dự Đoán (Time Update)

Ở bước này, chúng ta dự đoán trạng thái và hiệp phương sai sai số tại thời điểm $k$ dựa trên thông tin từ thời điểm $k-1$.
Giả sử chúng ta đã có ước lượng hậu nghiệm $\hat{\mathbf{x}}_{k-1|k-1}$ và hiệp phương sai $\mathbf{P}_{k-1|k-1}$ từ bước trước.

#### a. Dự đoán trạng thái tiên nghiệm ($\hat{\mathbf{x}}_{k|k-1}$)

Chúng ta sử dụng mô hình quá trình để dự đoán trạng thái:

$$ \hat{\mathbf{x}}_{k|k-1} = \mathbf{A}_k \hat{\mathbf{x}}_{k-1|k-1} + \mathbf{B}_k \mathbf{u}_k $$

*Chứng minh:*
Lấy kỳ vọng của phương trình mô hình quá trình $\mathbf{x}_k = \mathbf{A}_k \mathbf{x}_{k-1} + \mathbf{B}_k \mathbf{u}_k + \mathbf{w}_{k-1}$, có điều kiện là tất cả các phép đo đến $k-1$ (ký hiệu $Z_{k-1}$):
$E[\mathbf{x}_k | Z_{k-1}] = E[\mathbf{A}_k \mathbf{x}_{k-1} + \mathbf{B}_k \mathbf{u}_k + \mathbf{w}_{k-1} | Z_{k-1}]$
$\hat{\mathbf{x}}_{k|k-1} = \mathbf{A}_k E[\mathbf{x}_{k-1} | Z_{k-1}] + \mathbf{B}_k \mathbf{u}_k + E[\mathbf{w}_{k-1} | Z_{k-1}]$
Vì $\mathbf{w}_{k-1}$ là nhiễu trắng có kỳ vọng bằng 0 và độc lập với các phép đo quá khứ, $E[\mathbf{w}_{k-1} | Z_{k-1}] = E[\mathbf{w}_{k-1}] = \mathbf{0}$.
Do đó: $\hat{\mathbf{x}}_{k|k-1} = \mathbf{A}_k \hat{\mathbf{x}}_{k-1|k-1} + \mathbf{B}_k \mathbf{u}_k$.

#### b. Dự đoán hiệp phương sai sai số tiên nghiệm ($\mathbf{P}_{k|k-1}$)

Sai số dự đoán tiên nghiệm là:
$\mathbf{e}_{k|k-1} = \mathbf{x}_k - \hat{\mathbf{x}}_{k|k-1}$.
$\mathbf{e}_{k|k-1} = (\mathbf{A}_k \mathbf{x}_{k-1} + \mathbf{B}_k \mathbf{u}_k + \mathbf{w}_{k-1}) - (\mathbf{A}_k \hat{\mathbf{x}}_{k-1|k-1} + \mathbf{B}_k \mathbf{u}_k)$
$\mathbf{e}_{k|k-1} = \mathbf{A}_k (\mathbf{x}_{k-1} - \hat{\mathbf{x}}_{k-1|k-1}) + \mathbf{w}_{k-1} = \mathbf{A}_k \mathbf{e}_{k-1|k-1} + \mathbf{w}_{k-1}$.

Hiệp phương sai:
$\mathbf{P}_{k|k-1} = E[\mathbf{e}_{k|k-1} \mathbf{e}_{k|k-1}^T]$
$= E[(\mathbf{A}_k \mathbf{e}_{k-1|k-1} + \mathbf{w}_{k-1})(\mathbf{A}_k \mathbf{e}_{k-1|k-1} + \mathbf{w}_{k-1})^T]$
$= E[\mathbf{A}_k \mathbf{e}_{k-1|k-1} \mathbf{e}_{k-1|k-1}^T \mathbf{A}_k^T + \mathbf{A}_k \mathbf{e}_{k-1|k-1} \mathbf{w}_{k-1}^T + \mathbf{w}_{k-1} \mathbf{e}_{k-1|k-1}^T \mathbf{A}_k^T + \mathbf{w}_{k-1} \mathbf{w}_{k-1}^T]$.

Vì sai số $\mathbf{e}_{k-1|k-1}$ và nhiễu quá trình $\mathbf{w}_{k-1}$ là độc lập và có kỳ vọng bằng không, các số hạng chéo (cross terms) sẽ bằng không khi lấy kỳ vọng.
$E[\mathbf{A}_k \mathbf{e}_{k-1|k-1} \mathbf{w}_{k-1}^T] = \mathbf{A}_k E[\mathbf{e}_{k-1|k-1}] E[\mathbf{w}_{k-1}^T] = \mathbf{0}$ (vì $E[\mathbf{e}_{k-1|k-1}] = \mathbf{0}$ nếu ước lượng là không chệch).
Do đó:
$\mathbf{P}_{k|k-1} = \mathbf{A}_k E[\mathbf{e}_{k-1|k-1} \mathbf{e}_{k-1|k-1}^T] \mathbf{A}_k^T + E[\mathbf{w}_{k-1} \mathbf{w}_{k-1}^T]$

$$ \mathbf{P}_{k|k-1} = \mathbf{A}_k \mathbf{P}_{k-1|k-1} \mathbf{A}_k^T + \mathbf{Q}_{k-1} $$

(Lưu ý: một số tài liệu dùng $\mathbf{Q}_k$ thay vì $\mathbf{Q}_{k-1}$, tùy thuộc vào quy ước chỉ số thời gian của nhiễu).

### 2. Bước Hiệu Chỉnh (Measurement Update)

Khi có phép đo mới $\mathbf{z}_k$, chúng ta sử dụng nó để hiệu chỉnh ước lượng tiên nghiệm $\hat{\mathbf{x}}_{k|k-1}$ thành ước lượng hậu nghiệm $\hat{\mathbf{x}}_{k|k}$.
Ý tưởng cơ bản là kết hợp tuyến tính giữa dự đoán tiên nghiệm và thông tin từ phép đo mới:

$$ \hat{\mathbf{x}}_{k|k} = (1-\alpha) \hat{\mathbf{x}}_{k|k-1} + \alpha \cdot (\text{thông tin từ } \mathbf{z}_k) $$

Trong bộ lọc Kalman, dạng cụ thể hơn là:

$$ \hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k (\mathbf{z}_k - \mathbf{H}_k \hat{\mathbf{x}}_{k|k-1}) $$
 
Term $(\mathbf{z}_k - \mathbf{H}_k \hat{\mathbf{x}}_{k|k-1})$ được gọi là **innovation** hoặc **measurement residual** (phần dư đo lường), ký hiệu là $\tilde{\mathbf{y}}_k$. Nó biểu diễn sự khác biệt giữa phép đo thực tế $\mathbf{z}_k$ và phép đo dự kiến $\mathbf{H}_k \hat{\mathbf{x}}_{k|k-1}$ (dựa trên dự đoán trạng thái tiên nghiệm).
Ma trận $\mathbf{K}_k$ được gọi là **Kalman Gain**. Nó quyết định mức độ chúng ta "tin tưởng" vào phép đo mới (innovation) so với dự đoán tiên nghiệm. Nếu $\mathbf{K}_k$ nhỏ, chúng ta tin vào dự đoán nhiều hơn. Nếu $\mathbf{K}_k$ lớn, chúng ta tin vào phép đo nhiều hơn.

#### a. Tính toán Kalman Gain ($\mathbf{K}_k$)

Mục tiêu là chọn $\mathbf{K}_k$ sao cho phương sai của sai số ước lượng hậu nghiệm $\mathbf{P}_{k|k}$ là nhỏ nhất. Điều này tương đương với việc tối thiểu hóa trace (tổng các phần tử trên đường chéo chính) của $\mathbf{P}_{k|k}$.

Sai số hậu nghiệm: $\mathbf{e}_{k|k} = \mathbf{x}_k - \hat{\mathbf{x}}_{k|k}$

$$\mathbf{e}_{k|k} = \mathbf{x}_k - [\hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k (\mathbf{z}_k - \mathbf{H}_k \hat{\mathbf{x}}_{k|k-1})]$$

Thay $\mathbf{z}_k = \mathbf{H}_k \mathbf{x}_k + \mathbf{v}_k$:

$$\mathbf{e}_{k|k} = \mathbf{x}_k - \hat{\mathbf{x}}_{k|k-1} - \mathbf{K}_k (\mathbf{H}_k \mathbf{x}_k + \mathbf{v}_k - \mathbf{H}_k \hat{\mathbf{x}}_{k|k-1})$$

$$\mathbf{e}_{k|k} = (\mathbf{x}_k - \hat{\mathbf{x}}_{k|k-1}) - \mathbf{K}_k \mathbf{H}_k (\mathbf{x}_k - \hat{\mathbf{x}}_{k|k-1}) - \mathbf{K}_k \mathbf{v}_k$$

$$\mathbf{e}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{e}_{k|k-1} - \mathbf{K}_k \mathbf{v}_k$$

Hiệp phương sai sai số hậu nghiệm: $\mathbf{P}_{k|k} = E[\mathbf{e}_{k|k} \mathbf{e}_{k|k}^T]$

$$\mathbf{P}_{k|k} = E[ ((\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{e}_{k|k-1} - \mathbf{K}_k \mathbf{v}_k) ((\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{e}_{k|k-1} - \mathbf{K}_k \mathbf{v}_k)^T ]$$

Vì $\mathbf{e}_{k|k-1}$ (dựa trên $\mathbf{w}$ đến $k-1$) và $\mathbf{v}_k$ (nhiễu đo lường tại $k$) là độc lập và có kỳ vọng bằng không, các số hạng chéo sẽ bằng không:

$$\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) E[\mathbf{e}_{k|k-1} \mathbf{e}_{k|k-1}^T] (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k)^T + \mathbf{K}_k E[\mathbf{v}_k \mathbf{v}_k^T] \mathbf{K}_k^T$$

$$\Leftrightarrow \mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_{k|k-1} (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k)^T + \mathbf{K}_k \mathbf{R}_k \mathbf{K}_k^T$$

Đây là **dạng Joseph (Joseph form)** của phương trình cập nhật hiệp phương sai, nó ổn định hơn về mặt số học.

Để tìm $\mathbf{K}_k$ tối ưu, ta lấy đạo hàm của trace của $\mathbf{P}_{k|k}$ theo $\mathbf{K}_k$ và đặt bằng 0.

$$ \text{trace}(\mathbf{P}_{k|k}) = \text{trace}[(\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_{k|k-1} (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k)^T + \mathbf{K}_k \mathbf{R}_k \mathbf{K}_k^T] $$

$$\Leftrightarrow \text{trace}(\mathbf{P}_{k|k}) = \text{trace}[\mathbf{P}_{k|k-1} - \mathbf{K}_k \mathbf{H}_k \mathbf{P}_{k|k-1} - \mathbf{P}_{k|k-1} \mathbf{H}_k^T \mathbf{K}_k^T + \mathbf{K}_k \mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^T \mathbf{K}_k^T + \mathbf{K}_k \mathbf{R}_k \mathbf{K}_k^T]$$

Lấy đạo hàm theo $\mathbf{K}_k$ (sử dụng các quy tắc đạo hàm ma trận) ta có:

$$\frac{\partial \text{trace}(\mathbf{P}_{k|k})}{\partial \mathbf{K}_k} = -2 \mathbf{P}_{k|k-1} \mathbf{H}_k^T + 2 \mathbf{K}_k (\mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^T + \mathbf{R}_k) = \mathbf{0}$$

Giải phương trình này cho $\mathbf{K}_k$ ta được:

$$2 \mathbf{K}_k (\mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^T + \mathbf{R}_k) = 2 \mathbf{P}_{k|k-1} \mathbf{H}_k^T$$

$$\Rightarrow \mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}_k^T (\mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^T + \mathbf{R}_k)^{-1}$$

Ma trận $(\mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^T + \mathbf{R}_k)$ được gọi là **ma trận hiệp phương sai innovation (innovation covariance matrix)**, ký hiệu $\mathbf{S}_k$.

Vậy:

<div align="center">

$$
\boxed{\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}_k^T \mathbf{S}_k^{-1}}
$$

</div>

#### b. Cập nhật ước lượng trạng thái hậu nghiệm ($\hat{\mathbf{x}}_{k|k}$)

Như đã nêu ở trên:

$$ \hat{\mathbf{x}}_{k|k} = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k (\mathbf{z}_k - \mathbf{H}_k \hat{\mathbf{x}}_{k|k-1}) $$

#### c. Cập nhật hiệp phương sai sai số hậu nghiệm ($\mathbf{P}_{k|k}$)

Thay $\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}_k^T \mathbf{S}_k^{-1}$ vào dạng Joseph của $\mathbf{P}_{k|k}$ ở trên là một cách.
Một dạng phổ biến và đơn giản hơn (dù có thể kém ổn định số học hơn dạng Joseph trong một số trường hợp):
$\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_{k|k-1} (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k)^T + \mathbf{K}_k \mathbf{R}_k \mathbf{K}_k^T$
$= \mathbf{P}_{k|k-1} - \mathbf{K}_k \mathbf{H}_k \mathbf{P}_{k|k-1} - \mathbf{P}_{k|k-1} \mathbf{H}_k^T \mathbf{K}_k^T + \mathbf{K}_k \mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^T \mathbf{K}_k^T + \mathbf{K}_k \mathbf{R}_k \mathbf{K}_k^T$
$= \mathbf{P}_{k|k-1} - \mathbf{K}_k \mathbf{H}_k \mathbf{P}_{k|k-1} - \mathbf{P}_{k|k-1} \mathbf{H}_k^T \mathbf{K}_k^T + \mathbf{K}_k (\mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^T + \mathbf{R}_k) \mathbf{K}_k^T$
Từ định nghĩa $\mathbf{K}_k$, ta có $\mathbf{K}_k (\mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^T + \mathbf{R}_k) = \mathbf{P}_{k|k-1} \mathbf{H}_k^T$.
Vậy $(\mathbf{H}_k \mathbf{P}_{k|k-1} \mathbf{H}_k^T + \mathbf{R}_k) \mathbf{K}_k^T = \mathbf{H}_k \mathbf{P}_{k|k-1}$.
Thay vào:
$\mathbf{P}_{k|k} = \mathbf{P}_{k|k-1} - \mathbf{K}_k \mathbf{H}_k \mathbf{P}_{k|k-1}$
$= (\mathbf{I} - \mathbf{K}_k \mathbf{H}_k) \mathbf{P}_{k|k-1}$. Đây là dạng đơn giản và thường được sử dụng.

### 3. Khởi Tạo

Để bắt đầu chu trình đệ quy, chúng ta cần cung cấp:
*   **Ước lượng trạng thái ban đầu:** $\hat{\mathbf{x}}_{0|0}$
*   **Ma trận hiệp phương sai sai số ban đầu:** $\mathbf{P}_{0|0}$

Nếu chúng ta không có thông tin chính xác về trạng thái ban đầu, $\hat{\mathbf{x}}_{0|0}$ có thể được đặt là một giá trị hợp lý (ví dụ, dựa trên phép đo đầu tiên hoặc bằng không). $\mathbf{P}_{0|0}$ nên được đặt với các giá trị lớn trên đường chéo chính (thể hiện độ không chắc chắn ban đầu cao), và các giá trị ngoài đường chéo bằng không nếu không có thông tin về tương quan sai số ban đầu. Bộ lọc Kalman có xu hướng hội tụ về ước lượng đúng ngay cả khi giá trị khởi tạo không quá chính xác, miễn là hệ thống có thể quan sát được (observable).

---

## Phần IV: Kết Luận

<p style="text-align:center; font-weight:bold; font-size:1.2em; font-style:italic; border-radius:8px; padding:16px;">
  Bộ lọc Kalman là một công cụ mạnh mẽ cho ước lượng trạng thái tối ưu trong các hệ thống động tuyến tính với nhiễu Gaussian...
</p>

---
## Phần V: Ví Dụ Minh Họa - Theo dõi Chuyển Động 1D

Hãy xem xét một ví dụ theo dõi vị trí và vận tốc của một đối tượng chuyển động trên một đường thẳng với gia tốc ngẫu nhiên nhỏ.

*   **Trạng thái:** $\mathbf{x}_k = \begin{bmatrix} p_k \\ v_k \end{bmatrix}$, với $p_k$ là vị trí và $v_k$ là vận tốc tại thời điểm $k$.
*   **Mô hình quá trình:**
    Giả sử chuyển động với vận tốc gần như không đổi trong khoảng thời gian nhỏ $\Delta t$, nhưng có thể bị tác động bởi gia tốc ngẫu nhiên.
    $p_k = p_{k-1} + v_{k-1} \Delta t + \frac{1}{2} a_{k-1} (\Delta t)^2$
    $v_k = v_{k-1} + a_{k-1} \Delta t$
    Nhiễu quá trình $\mathbf{w}_{k-1}$ liên quan đến gia tốc ngẫu nhiên $a_{k-1}$.
    Ma trận $\mathbf{A} = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix}$.
    Ma trận $\mathbf{B}$ (nếu có điều khiển) hoặc cách mô hình hóa nhiễu quá trình $\mathbf{Q}$ sẽ phụ thuộc vào giả định về $a_{k-1}$. Một cách phổ biến là mô hình hóa nhiễu gia tốc liên tục:
    $\mathbf{Q} = \begin{bmatrix} \frac{\Delta t^4}{4} & \frac{\Delta t^3}{2} \\ \frac{\Delta t^3}{2} & \Delta t^2 \end{bmatrix} \sigma_a^2$, với $\sigma_a^2$ là phương sai của nhiễu gia tốc.
*   **Mô hình đo lường:** Giả sử chúng ta chỉ đo được vị trí $p_k$ với nhiễu.
    $z_k = p_k + v_k'$
    Ma trận $\mathbf{H} = \begin{bmatrix} 1 & 0 \end{bmatrix}$.
    Ma trận $\mathbf{R} = [\sigma_z^2]$ (scalar trong trường hợp này, với $\sigma_z^2$ là phương sai nhiễu đo vị trí).


#### Code Python (NumPy)

(Phần code Python bạn cung cấp cho ví dụ theo dõi vị trí 1D là rất tốt và trực quan. Tôi sẽ giữ nguyên nó và chỉ bổ sung một vài chú thích để làm rõ hơn mối liên hệ với các công thức đã chứng minh).

```python
import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, A, B, H, Q, R, x0, P0):
        self.A = A  # Ma trận chuyển đổi trạng thái
        self.B = B  # Ma trận điều khiển (có thể là None nếu không có u_k)
        self.H = H  # Ma trận đo lường
        self.Q = Q  # Hiệp phương sai nhiễu quá trình (cho wk)
        self.R = R  # Hiệp phương sai nhiễu đo lường (cho vk)
        
        self.x_hat = x0  # Ước lượng trạng thái ban đầu (x_0|0)
        self.P = P0    # Hiệp phương sai sai số ban đầu (P_0|0)
        
    def predict(self, u=None):
        """Bước Dự Đoán (Time Update)"""
        # Dự đoán trạng thái tiên nghiệm: x_hat_k|k-1 = A @ x_hat_k-1|k-1 (+ B @ u_k)
        if self.B is not None and u is not None:
            self.x_hat_apriori = self.A @ self.x_hat + self.B @ u
        else:
            self.x_hat_apriori = self.A @ self.x_hat
            
        # Dự đoán hiệp phương sai sai số tiên nghiệm: P_k|k-1 = A @ P_k-1|k-1 @ A.T + Q
        self.P_apriori = self.A @ self.P @ self.A.T + self.Q
        
        # Lưu lại để dùng ở bước cập nhật
        self.x_hat = self.x_hat_apriori 
        self.P = self.P_apriori
        
    def update(self, z):
        """Bước Hiệu Chỉnh (Measurement Update)"""
        # Tính innovation (measurement residual): y_tilde_k = z_k - H @ x_hat_k|k-1
        y_tilde = z - self.H @ self.x_hat_apriori
        
        # Tính hiệp phương sai innovation: S_k = H @ P_k|k-1 @ H.T + R
        S = self.H @ self.P_apriori @ self.H.T + self.R
        
        # Tính Kalman Gain: K_k = P_k|k-1 @ H.T @ inv(S_k)
        K = self.P_apriori @ self.H.T @ np.linalg.inv(S)
        
        # Cập nhật ước lượng trạng thái hậu nghiệm: x_hat_k|k = x_hat_k|k-1 + K_k @ y_tilde_k
        self.x_hat = self.x_hat_apriori + K @ y_tilde
        
        # Cập nhật hiệp phương sai sai số hậu nghiệm: P_k|k = (I - K_k @ H) @ P_k|k-1
        # (Hoặc dạng Joseph để ổn định hơn: P_k|k = (I - K_k @ H) @ P_k|k-1 @ (I - K_k @ H).T + K @ R @ K.T)
        I = np.eye(self.P_apriori.shape[0])
        self.P = (I - K @ self.H) @ self.P_apriori
        # self.P = (I - K @ self.H) @ self.P_apriori @ (I - K @ self.H).T + K @ self.R @ K.T # Dạng Joseph


# Tham số mô phỏng cho theo dõi vị trí và vận tốc 1D
dt = 0.1  # Bước thời gian (s)
n_steps = 200 # Số bước mô phỏng
process_noise_std = 0.2  # Độ lệch chuẩn của nhiễu gia tốc (m/s^2)
measurement_noise_std = 1.0 # Độ lệch chuẩn của nhiễu đo vị trí (m)

# Tạo dữ liệu thực (True states)
np.random.seed(42)
true_positions = np.zeros(n_steps)
true_velocities = np.zeros(n_steps)
measurements = np.zeros(n_steps)

# Giá trị ban đầu của hệ thống thực
current_pos = 0.0
current_vel = 1.0 # m/s

for i in range(n_steps):
    # Mô phỏng gia tốc ngẫu nhiên (nhiễu quá trình)
    random_acc = np.random.normal(0, process_noise_std)
    
    # Cập nhật trạng thái thực
    current_pos = current_pos + current_vel * dt + 0.5 * random_acc * dt**2
    current_vel = current_vel + random_acc * dt
    
    true_positions[i] = current_pos
    true_velocities[i] = current_vel
    
    # Tạo phép đo với nhiễu
    measurements[i] = current_pos + np.random.normal(0, measurement_noise_std)

# Khởi tạo các ma trận cho Bộ lọc Kalman
# Mô hình quá trình: x_k = A * x_{k-1} + w_{k-1} (không có điều khiển B*u)
# x = [position, velocity]^T
A = np.array([[1, dt],
              [0, 1]])

# Ma trận H cho mô hình đo lường: z_k = H * x_k + v_k
# Chúng ta chỉ đo vị trí
H = np.array([[1, 0]])

# Ma trận hiệp phương sai nhiễu quá trình Q
# Mô hình hóa nhiễu do gia tốc ngẫu nhiên tác động lên cả vị trí và vận tốc
# Q = G * G.T * sigma_a^2, với G = [dt^2/2, dt]^T
# Đây là một cách xây dựng Q phổ biến cho mô hình gia tốc không đổi bị nhiễu.
G = np.array([[0.5*dt**2], [dt]])
Q = G @ G.T * (process_noise_std**2)
# Hoặc một dạng đơn giản hơn nếu giả định nhiễu độc lập cho vị trí và vận tốc
# Q = np.array([[ (dt**4)/4 * process_noise_std**2 , (dt**3)/2 * process_noise_std**2],
#               [ (dt**3)/2 * process_noise_std**2 ,  dt**2 * process_noise_std**2 ]])

# Ma trận hiệp phương sai nhiễu đo lường R
R = np.array([[measurement_noise_std**2]])

# Điều kiện ban đầu cho bộ lọc
x0_hat = np.array([0.0, 0.0])  # Ước lượng ban đầu về vị trí và vận tốc
P0 = np.array([[1.0, 0.0],       # Phương sai vị trí ban đầu lớn
               [0.0, 1.0]])      # Phương sai vận tốc ban đầu lớn

kf = KalmanFilter(A=A, B=None, H=H, Q=Q, R=R, x0=x0_hat, P0=P0)

# Chạy bộ lọc qua các bước thời gian
estimated_positions_kf = []
estimated_velocities_kf = []
P_diagonal_variances = [] # Lưu trữ phương sai (đường chéo của P)

for i in range(n_steps):
    # Bước dự đoán
    kf.predict() 
    
    # Bước cập nhật với phép đo mới
    kf.update(np.array([measurements[i]]))
    
    estimated_positions_kf.append(kf.x_hat[0])
    estimated_velocities_kf.append(kf.x_hat[1])
    P_diagonal_variances.append(np.diag(kf.P))


estimated_positions_kf = np.array(estimated_positions_kf)
estimated_velocities_kf = np.array(estimated_velocities_kf)
P_diagonal_variances = np.array(P_diagonal_variances)

# Tính toán khoảng tin cậy (ví dụ 95% CI là +/- 1.96 * std_dev)
pos_std_dev = np.sqrt(P_diagonal_variances[:, 0])
vel_std_dev = np.sqrt(P_diagonal_variances[:, 1])

# Trực quan hóa kết quả
time_axis = np.arange(n_steps) * dt

plt.figure(figsize=(16, 12))

plt.subplot(2, 1, 1)
plt.plot(time_axis, true_positions, 'g-', label='Vị trí Thực', linewidth=2, alpha=0.8)
plt.plot(time_axis, measurements, 'ro', label='Đo lường Thô', markersize=4, alpha=0.5)
plt.plot(time_axis, estimated_positions_kf, 'b-', label='Ước lượng Vị trí (KF)', linewidth=2)
plt.fill_between(time_axis, 
                 estimated_positions_kf - 1.96 * pos_std_dev,
                 estimated_positions_kf + 1.96 * pos_std_dev,
                 color='blue', alpha=0.2, label='95% Khoảng Tin Cậy (Vị trí)')
plt.xlabel('Thời gian (s)')
plt.ylabel('Vị trí (m)')
plt.title('Theo dõi Vị trí 1D với Bộ lọc Kalman')
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(2, 1, 2)
plt.plot(time_axis, true_velocities, 'g-', label='Vận tốc Thực', linewidth=2, alpha=0.8)
plt.plot(time_axis, estimated_velocities_kf, 'b-', label='Ước lượng Vận tốc (KF)', linewidth=2)
plt.fill_between(time_axis, 
                 estimated_velocities_kf - 1.96 * vel_std_dev,
                 estimated_velocities_kf + 1.96 * vel_std_dev,
                 color='blue', alpha=0.2, label='95% Khoảng Tin Cậy (Vận tốc)')
plt.xlabel('Thời gian (s)')
plt.ylabel('Vận tốc (m/s)')
plt.title('Ước lượng Vận tốc 1D với Bộ lọc Kalman')
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Tính toán lỗi RMS (Root Mean Square Error)
position_rmse_kf = np.sqrt(np.mean((estimated_positions_kf - true_positions)**2))
velocity_rmse_kf = np.sqrt(np.mean((estimated_velocities_kf - true_velocities)**2))
raw_measurement_rmse = np.sqrt(np.mean((measurements - true_positions)**2))

print(f"RMSE Vị trí (Bộ lọc Kalman): {position_rmse_kf:.4f} m")
print(f"RMSE Vận tốc (Bộ lọc Kalman): {velocity_rmse_kf:.4f} m/s")
print(f"RMSE Đo lường Thô (so với vị trí thực): {raw_measurement_rmse:.4f} m")
```
