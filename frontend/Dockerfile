FROM node:20-slim

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

# Build Vite app
RUN npm run build

EXPOSE 5173
# For a student project, serving via vite dev is easiest for testing
CMD ["npm", "run", "dev", "--", "--host"]