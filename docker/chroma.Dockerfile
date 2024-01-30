FROM chromadb/chroma:latest

ENV CHROMA_SERVER_AUTH_CREDENTIALS_FILE="server.htpasswd"
ENV CHROMA_SERVER_AUTH_CREDENTIALS_PROVIDER='chromadb.auth.providers.HtpasswdFileServerAuthCredentialsProvider'
ENV CHROMA_SERVER_AUTH_PROVIDER='chromadb.auth.basic.BasicAuthServerProvider'

RUN apt update -y && apt install -y apache2-utils
RUN htpasswd -Bbn local local > server.htpasswd

EXPOSE 8000
