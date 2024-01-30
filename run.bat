::docker run --name pgvector -p 5432:5432 -v %CD%/db:/var/lib/postgresql/data/pgdata  -d pgvector:latest
::pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
docker run -d --name chroma-test -v %CD%/chroma:/chroma/chroma -p 8000:8000 chroma-local