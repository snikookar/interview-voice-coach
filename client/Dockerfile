# Interview Voice Coach: Next.js client.
# docker build -t voice-coach-client --build-arg API_URL=http://127.0.0.1:7860 client/
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# Rewrites (/api -> FastAPI) are resolved at build time.
ARG API_URL=http://127.0.0.1:7860
ENV API_URL=$API_URL NEXT_TELEMETRY_DISABLED=1
RUN npm run build

FROM node:22-alpine
WORKDIR /app
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1
COPY --from=build /app/package.json /app/package-lock.json ./
RUN npm ci --omit=dev
COPY --from=build /app/.next ./.next
COPY --from=build /app/public ./public
COPY --from=build /app/next.config.ts ./
EXPOSE 3000
CMD ["npx", "next", "start", "-p", "3000"]
