# Use an official Node runtime as the base image  
FROM node:18-alpine  

# Set the working directory in the container to /app  
WORKDIR /app  

# Copy package.json and package-lock.json  
COPY package*.json ./  

# Install app dependencies  
RUN npm install  

# Bundle app source  
COPY . .  

# Build the app  
RUN npm run build  

# The app listens on port 3000, so expose this port  
EXPOSE 3000  

# Define the command to run the app  
CMD [ "npm", "start" ]  