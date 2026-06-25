import basicSsl from "@vitejs/plugin-basic-ssl";
import { defineConfig } from "vite";
import { createReadStream, existsSync, statSync } from "node:fs";
import path from "node:path";

const repoRoot = process.cwd();

function contentType(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  if (ext === ".wasm") return "application/wasm";
  if (ext === ".js") return "text/javascript; charset=utf-8";
  if (ext === ".tflite") return "application/octet-stream";
  if (ext === ".png") return "image/png";
  if (ext === ".jpg" || ext === ".jpeg") return "image/jpeg";
  return "application/octet-stream";
}

function staticFiles(prefix, directory) {
  const root = path.resolve(repoRoot, directory);
  return {
    name: `serve-${prefix}`,
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const requestUrl = req.url?.split("?")[0] ?? "";
        if (!requestUrl.startsWith(prefix)) {
          next();
          return;
        }

        const relativePath = decodeURIComponent(requestUrl.slice(prefix.length));
        const filePath = path.resolve(root, relativePath);
        if (!filePath.startsWith(root) || !existsSync(filePath) || !statSync(filePath).isFile()) {
          res.statusCode = 404;
          res.end("Not found");
          return;
        }

        res.setHeader("Content-Type", contentType(filePath));
        createReadStream(filePath).pipe(res);
      });
    },
  };
}

export default defineConfig({
  plugins: [
    basicSsl(),
    staticFiles("/litert-wasm/", "node_modules/@litertjs/core/wasm"),
    staticFiles("/tflite/", "tflite"),
    staticFiles("/demo_images/", "demo_images"),
  ],
  server: {
    headers: {
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Opener-Policy": "same-origin",
    },
  },
});
