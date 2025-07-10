# 🚀 Nguyen Blog

Blog cá nhân được xây dựng bằng [Astro](https://astro.build) - nơi chia sẻ kiến thức về lập trình, công nghệ và cuộc sống.  
Dự án này được fork từ [saicaca/fuwari](https://github.com/saicaca/fuwari.git).

![Node.js >= 20](https://img.shields.io/badge/node.js-%3E%3D20-brightgreen) 
![pnpm >= 9](https://img.shields.io/badge/pnpm-%3E%3D9-blue) 

[**🖥️ Live Demo (Vercel)**](https://nguyen-blog.vercel.app)

## ✨ Tính năng

- [x] Được xây dựng bằng [Astro](https://astro.build) và [Tailwind CSS](https://tailwindcss.com)
- [x] Giao diện responsive và hiện đại
- [x] Chế độ sáng / tối
- [x] Màu sắc và banner có thể tùy chỉnh
- [x] Tìm kiếm với [Pagefind](https://pagefind.app/)
- [x] [Markdown mở rộng](https://github.com/saicaca/fuwari?tab=readme-ov-file#-markdown-extended-syntax)
- [x] Mục lục tự động
- [x] RSS feed
- [x] Hỗ trợ tiếng Việt

## 🚀 Bắt đầu

1. Clone repository này:
   ```sh
   git clone https://github.com/vtnguyen04/NguyenBlog.git
   cd NguyenBlog
   ```

2. Cài đặt dependencies:
   ```sh
   pnpm install
   ```

3. Chạy development server:
   ```sh
   pnpm dev
   ```

4. Tùy chỉnh cấu hình trong `src/config.ts`

5. Tạo bài viết mới:
   ```sh
   pnpm new-post <tên-file>
   ```

## 📝 Cấu trúc bài viết

```yaml
---
title: "Tiêu đề bài viết"
published: 2024-01-15
description: "Mô tả ngắn gọn về bài viết"
image: ./cover.jpg
tags: [Tag1, Tag2]
category: "Lập trình"
draft: false
lang: "vi"
---
```

## 🧩 Markdown mở rộng

Blog hỗ trợ các tính năng Markdown mở rộng:

- Admonitions (Ghi chú, cảnh báo, tips)
- GitHub repository cards
- Code blocks với syntax highlighting
- Math equations với KaTeX

## ⚡ Lệnh hữu ích

| Lệnh                    | Mô tả                                              |
|:---------------------------|:----------------------------------------------------|
| `pnpm install`             | Cài đặt dependencies                               |
| `pnpm dev`                 | Chạy server development tại `localhost:4321`         |
| `pnpm build`               | Build website production vào `./dist/`             |
| `pnpm preview`             | Xem trước build trước khi deploy                   |
| `pnpm check`               | Kiểm tra lỗi trong code                            |
| `pnpm format`              | Format code bằng Biome                             |
| `pnpm new-post <filename>` | Tạo bài viết mới                                   |

## 📄 License

Dự án này được cấp phép theo MIT License.

---

**Nguyen Blog** - Nơi chia sẻ kiến thức và trải nghiệm về lập trình, công nghệ và cuộc sống. 🎉
