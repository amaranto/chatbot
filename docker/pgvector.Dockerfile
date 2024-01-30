FROM ankane/pgvector:latest

ENV PGDATA=/var/lib/postgresql/data/pgdata
ENV POSTGRES_USER=local
ENV POSTGRES_PASSWORD=local
ENV POSTGRES_DB=local

EXPOSE 5432 