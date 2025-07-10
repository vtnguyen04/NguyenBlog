# Hướng dẫn sử dụng Nguyen Blog

## 🚀 Cách chạy blog

### 1. Cài đặt dependencies
```bash
pnpm install
```

### 2. Chạy development server
```bash
pnpm dev
```
Blog sẽ chạy tại `http://localhost:4321`

### 3. Build production
```bash
pnpm build
```

## 📝 Cách tạo bài viết mới

### Sử dụng script có sẵn:
```bash
pnpm new-post "tên-bài-viết"
```

### Hoặc tạo file thủ công:
Tạo file `.md` trong thư mục `src/content/posts/` với cấu trúc:

```yaml
---
title: "Tiêu đề bài viết"
published: 2024-01-15
description: "Mô tả ngắn gọn về bài viết"
image: "./cover.jpg"  # Ảnh đại diện (tùy chọn)
tags: [Tag1, Tag2, Tag3]
category: "Lập trình"
draft: false
lang: "vi"
---

# Nội dung bài viết

Viết nội dung bằng Markdown...
```

## ⚙️ Tùy chỉnh cấu hình

### 1. Thông tin cá nhân
Chỉnh sửa file `src/config.ts`:

```typescript
export const siteConfig: SiteConfig = {
  title: "Nguyen Blog",
  subtitle: "Blog cá nhân chia sẻ kiến thức và trải nghiệm",
  lang: "vi",
  // ...
};

export const profileConfig: ProfileConfig = {
  name: "Tên của bạn",
  bio: "Mô tả về bản thân",
  avatar: "assets/images/avatar.png",
  links: [
    {
      name: "GitHub",
      icon: "fa6-brands:github",
      url: "https://github.com/username",
    },
    // Thêm các link khác...
  ],
};
```

### 2. Navigation
Thêm/sửa các link trong navigation:

```typescript
export const navBarConfig: NavBarConfig = {
  links: [
    LinkPreset.Home,
    LinkPreset.Archive,
    LinkPreset.About,
    {
      name: "GitHub",
      url: "https://github.com/username",
      external: true,
    },
  ],
};
```

## 🎨 Tùy chỉnh giao diện

### 1. Màu sắc
Thay đổi màu chủ đạo trong `src/config.ts`:

```typescript
themeColor: {
  hue: 250, // Thay đổi giá trị từ 0-360
  fixed: false, // true để ẩn color picker
},
```

### 2. Banner
Thêm banner cho blog:

```typescript
banner: {
  enable: true,
  src: "assets/images/banner.png",
  position: "center", // 'top', 'center', 'bottom'
  credit: {
    enable: true,
    text: "Credit text",
    url: "https://example.com",
  },
},
```

## 📱 Deploy

### Deploy lên Vercel:
1. Push code lên GitHub
2. Kết nối repository với Vercel
3. Deploy tự động

### Deploy lên Netlify:
1. Push code lên GitHub
2. Kết nối repository với Netlify
3. Build command: `pnpm build`
4. Publish directory: `dist`

## 🔧 Các lệnh hữu ích

| Lệnh | Mô tả |
|------|-------|
| `pnpm dev` | Chạy development server |
| `pnpm build` | Build production |
| `pnpm preview` | Xem trước build |
| `pnpm new-post <tên>` | Tạo bài viết mới |
| `pnpm format` | Format code |
| `pnpm check` | Kiểm tra lỗi |

## 📚 Markdown Features

Blog hỗ trợ các tính năng Markdown mở rộng:

### 1. Admonitions
```markdown
:::note
Đây là một ghi chú
:::

:::warning
Đây là một cảnh báo
:::

:::tip
Đây là một tip
:::
```

### 2. Code blocks
```markdown
```javascript
console.log("Hello World!");
```
```

### 3. Math equations
```markdown
Inline: $E = mc^2$

Block:
$$
\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
$$
```

### 4. GitHub cards
```markdown
::github{repo="username/repo"}
```

## 🎯 Tips

1. **SEO**: Điền đầy đủ `description` cho mỗi bài viết
2. **Tags**: Sử dụng tags có ý nghĩa để dễ tìm kiếm
3. **Images**: Tối ưu hóa ảnh trước khi upload
4. **Draft**: Sử dụng `draft: true` để lưu bài viết nháp
5. **Categories**: Phân loại bài viết rõ ràng

## 🐛 Troubleshooting

### Lỗi build:
- Kiểm tra syntax trong Markdown
- Đảm bảo frontmatter đúng format
- Chạy `pnpm check` để kiểm tra lỗi

### Lỗi development server:
- Xóa thư mục `node_modules` và chạy lại `pnpm install`
- Kiểm tra port 4321 có đang được sử dụng không

---

**Chúc bạn có một blog tuyệt vời! 🎉** 