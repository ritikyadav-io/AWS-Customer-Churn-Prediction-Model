import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    {
      name: 'print-backend-links',
      configureServer(server) {
        server.httpServer?.once('listening', () => {
          setTimeout(() => {
            console.log('\n  \x1b[36m\x1b[1m🚀 ML PIPELINE SERVERS READY:\x1b[0m')
            console.log('  \x1b[32m➜  Local Frontend:\x1b[0m      \x1b[4mhttp://localhost:5173/\x1b[0m')
            console.log('  \x1b[33m➜  Flask ML Backend API:\x1b[0m \x1b[4mhttp://127.0.0.1:5000/\x1b[0m')
            console.log('  \x1b[35m➜  Model Predict Route:\x1b[0m  \x1b[4mhttp://127.0.0.1:5000/predict\x1b[0m\n')
          }, 100)
        })
      }
    }
  ],
  server: {
    port: 5173,
    host: true,
    proxy: {
      '/predict': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
      },
      '/api': {
        target: 'http://127.0.0.1:5000',
        changeOrigin: true,
      }
    }
  }
})
