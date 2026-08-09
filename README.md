# AI Face Service

Independent FastAPI service for face detection, registration, verification,
and closed-set identification. The service is built and tested independently
before integration with the Spring Boot attendance backend.

This service covers Phase 1 through Phase 10: setup, API design, detection,
aligned embeddings, registration, verification, threshold calibration, error
handling, performance measurement, and container deployment. It uses one
InsightFace pipeline for detection, alignment, and 512-dimensional embeddings,
with shared PostgreSQL persistence for Spring Boot integration and a SQLite
fallback for isolated development.

## Current scope

| Capability | Status |
| --- | --- |
| FastAPI application and environment configuration | Implemented |
| `GET /health` | Implemented |
| Register, verify, identify, and detect API contracts | Implemented and contract-tested |
| Upload type, empty-file, and size validation | Implemented |
| Consistent JSON error envelope | Implemented |
| InsightFace startup model loading | Implemented |
| Single-face detection and bounding box | Implemented |
| Aligned, normalized 512-D embedding | Implemented |
| Atomic registration/re-registration | Implemented |
| Shared PostgreSQL and local SQLite embedding persistence | Implemented |
| Cosine-similarity verification | Implemented |
| Candidate-restricted closed-set identification | Implemented |
| Labeled-pair threshold calibration | Implemented |
| Image and face quality checks | Implemented |
| Consistent errors including unexpected failures | Implemented |
| Bounded latency, CPU, and memory metrics | Implemented |
| Concurrent HTTP benchmark tool | Implemented |
| Docker and Compose deployment | Implemented |

Registration stores one current embedding per student. Registering the same
student again atomically replaces the previous embedding and returns a new
embedding ID. Verification compares normalized vectors using cosine similarity
and `SIMILARITY_THRESHOLD`.

## Project structure

```text
.
├── app/
│   ├── api/
│   │   └── routes.py
│   ├── models/
│   │   └── schemas.py
│   ├── repositories/
│   │   └── embedding_repository.py
│   ├── services/
│   │   ├── face_service.py
│   │   ├── insightface_service.py
│   │   ├── performance.py
│   │   └── threshold_tuning.py
│   ├── utils/
│   │   ├── images.py
│   │   └── uploads.py
│   ├── config.py
│   └── main.py
├── scripts/
│   ├── benchmark_api.py
│   └── tune_threshold.py
├── tests/
│   ├── test_api.py
│   ├── test_embedding_repository.py
│   ├── test_insightface_service.py
│   └── test_threshold_tuning.py
├── .env.example
├── compose.yaml
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Local setup

Use Python 3.11, matching the Docker image:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
cp .env.example .env
```

The development dependencies can run all API and pipeline unit tests without
downloading InsightFace weights. To run real inference, install the production
requirements:

```bash
python -m pip install -r requirements.txt
```

Run the service:

```bash
uvicorn app.main:app --reload
```

OpenAPI documentation is available at
`http://localhost:8000/docs`.

On first startup, InsightFace may download the configured model pack into
`INSIGHTFACE_MODEL_ROOT`. Startup does not complete until the detector and
recognition model are ready. For offline deployments, place the model files at
`<model-root>/models/<model-name>/` before starting the service.

## Tests

```bash
pytest
```

The tests inject a fake analyzer while exercising image decoding, model
preparation, exactly-one-face enforcement, bounding-box handling, embedding
dimension validation, L2 normalization, PostgreSQL/SQLite persistence,
replacement semantics, threshold metrics, quality rejection, error envelopes, threshold
matching, bounded performance tracking, benchmark summaries, and
unregistered-student errors. This keeps automated tests independent of a
large model download.

## API

### Health

```http
GET /health
```

```json
{
  "status": "UP"
}
```

Every HTTP response includes `X-Process-Time-Ms`, measuring server-side request
processing time.

### Performance metrics

```http
GET /metrics/performance
```

The response contains recent count/min/max/average/p50/p95/p99 timings and
current process resource data:

```json
{
  "uptimeSeconds": 120.5,
  "sampleLimitPerMetric": 1000,
  "metrics": {
    "insightface_analysis_ms": {
      "count": 20,
      "averageMs": 83.4,
      "p95Ms": 97.2
    }
  },
  "process": {
    "residentMemoryBytes": 734003200,
    "cpuPercent": 78.2,
    "threadCount": 18
  }
}
```

The example is abbreviated. Metrics are process-local and retained only in
memory; each metric keeps at most `PERFORMANCE_SAMPLE_LIMIT` recent samples.

### Register face

```http
POST /faces/register
Content-Type: multipart/form-data
```

Fields:

- `studentId`: non-empty student identifier
- `image`: JPEG, PNG, or WebP image

```json
{
  "success": true,
  "embeddingId": "emb-123"
}
```

Example:

```bash
curl -X POST http://localhost:8000/faces/register \
  -F 'studentId=STU-001' \
  -F 'image=@face.jpg'
```

Registration generates a normalized embedding and stores it in the configured
database. It does not store the uploaded image. Re-registering `STU-001`
replaces that student's previous embedding.

### Verify face

```http
POST /faces/verify
Content-Type: multipart/form-data
```

Fields:

- `studentId`: non-empty student identifier
- `image`: JPEG, PNG, or WebP image

```json
{
  "matched": true,
  "similarity": 0.84
}
```

Example:

```bash
curl -X POST http://localhost:8000/faces/verify \
  -F 'studentId=STU-001' \
  -F 'image=@verification-face.jpg'
```

`matched` is true when `similarity >= SIMILARITY_THRESHOLD`. An unknown
`studentId` returns HTTP `404` with `student_not_registered`.
Similarity below the threshold is a successful HTTP `200` response with
`"matched": false`; it is not an API error.

### Identify face among candidates

```http
POST /faces/identify
Content-Type: multipart/form-data
```

Fields:

- `image`: JPEG, PNG, or WebP image containing exactly one face
- `candidateStudentIds`: JSON-array string containing Spring student UUIDs

Example:

```bash
curl -X POST http://localhost:8000/faces/identify \
  -F 'candidateStudentIds=["2f52f06f-59ed-4519-bb86-69cb59fb3197","12807f44-e4e2-464a-b525-9812b3dc0f3c"]' \
  -F 'image=@attendance-face.jpg'
```

Matched response:

```json
{
  "matched": true,
  "studentId": "2f52f06f-59ed-4519-bb86-69cb59fb3197",
  "similarity": 0.93,
  "livenessPassed": true,
  "reason": "MATCHED"
}
```

The service generates one query embedding, loads registered embeddings only
for the supplied candidates, and returns the highest score. It never searches
students outside the supplied set. Unknown or unregistered candidate UUIDs are
ignored. When none of the registered candidates reaches
`SIMILARITY_THRESHOLD`, the response has `matched: false`, a null `studentId`,
and reason `NOT_MATCHED`.

No face and multiple-face inputs return HTTP `200` identification responses
with reasons `NO_FACE_DETECTED` and `MULTIPLE_FACES`, respectively. Invalid
candidate JSON, an empty candidate array, or more than
`MAX_IDENTIFY_CANDIDATES` entries returns HTTP `400`.

`livenessPassed` is currently always `true` as a bench-prototype compatibility
value. No liveness or anti-spoofing model is implemented, so this field must not
be treated as proof of a live person in production.

### Detect face

```http
POST /faces/detect
Content-Type: multipart/form-data
```

Field:

- `image`: JPEG, PNG, or WebP image

```json
{
  "faceFound": true,
  "boundingBox": {
    "x1": 10,
    "y1": 20,
    "x2": 110,
    "y2": 140
  },
  "confidence": 0.98
}
```

Detection requires exactly one face. Zero or multiple faces return HTTP `422`.

### Generate embedding

```http
POST /faces/embedding
Content-Type: multipart/form-data
```

Field:

- `image`: JPEG, PNG, or WebP image containing exactly one face

```json
{
  "dimension": 512,
  "embedding": [
    0.0123,
    -0.0345
  ]
}
```

The example vector is abbreviated. The actual response always contains 512
finite values and is normalized to unit length.

```bash
curl -X POST http://localhost:8000/faces/embedding \
  -F 'image=@face.jpg'
```

This endpoint exposes biometric template data for development verification.
Protect or disable it before exposing the service outside a trusted network.

### Error format

All application and validation errors use one envelope:

```json
{
  "error": {
    "code": "no_face_detected",
    "message": "No face was detected. Use a clear, front-facing image."
  }
}
```

| Status | Error codes |
| --- | --- |
| `400` | `empty_image`, `invalid_image` |
| `404` | `student_not_registered` |
| `413` | `image_too_large` |
| `415` | `unsupported_image_type` |
| `422` | `validation_error`, `invalid_student_id`, `no_face_detected`, `multiple_faces_detected`, `poor_image_quality` |
| `500` | `inference_failed`, `invalid_embedding`, `embedding_storage_failed`, `internal_error` |
| `503` | `pipeline_not_ready`, `pipeline_load_failed` |

Unexpected exceptions are logged server-side and returned as a generic
`internal_error`; implementation details are not exposed to API clients.

## Configuration

Settings are read from environment variables and, during local development,
from `.env`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `MAX_UPLOAD_SIZE_MB` | `10` | Maximum accepted image size |
| `SIMILARITY_THRESHOLD` | `0.50` | Minimum cosine similarity for a match |
| `MAX_IDENTIFY_CANDIDATES` | `500` | Maximum candidate UUIDs accepted by one identification request |
| `FACE_DATABASE_URL` | empty | PostgreSQL URL; when set, PostgreSQL replaces the SQLite fallback |
| `FACE_DATABASE_PATH` | `data/faces.db` | Local SQLite embedding database |
| `FACE_DATABASE_TIMEOUT_SECONDS` | `5.0` | Database connection/lock timeout |
| `INSIGHTFACE_MODEL_NAME` | `buffalo_l` | InsightFace model pack |
| `INSIGHTFACE_MODEL_ROOT` | `~/.insightface` | Parent directory for model files |
| `INSIGHTFACE_PROVIDERS` | `CPUExecutionProvider` | Comma-separated ONNX Runtime providers |
| `INSIGHTFACE_CONTEXT_ID` | `0` | InsightFace/ONNX execution context |
| `INSIGHTFACE_DETECTION_THRESHOLD` | `0.50` | Minimum detector confidence |
| `INSIGHTFACE_DETECTION_WIDTH` | `640` | Detection input width |
| `INSIGHTFACE_DETECTION_HEIGHT` | `640` | Detection input height |
| `MAX_IMAGE_PIXELS` | `20000000` | Decoded-image pixel safety limit |
| `MIN_IMAGE_WIDTH` | `160` | Minimum uploaded image width |
| `MIN_IMAGE_HEIGHT` | `160` | Minimum uploaded image height |
| `MIN_FACE_SIZE_PIXELS` | `80` | Minimum detected face width and height |
| `MIN_BLUR_SCORE` | `30.0` | Minimum variance-of-Laplacian sharpness score; `0` disables |
| `PERFORMANCE_SAMPLE_LIMIT` | `1000` | Recent timing samples retained per metric |

Do not store face images, embeddings, database credentials, or secrets in
`.env.example` or source control.

## Docker

The container runs as a non-root user, has a health check, uses one Uvicorn
worker, and writes only to the model/data volumes and `/tmp`.

Build and start with Compose:

```bash
docker compose build
docker compose up -d
docker compose ps
curl http://localhost:8000/health
```

Follow startup and model-download logs:

```bash
docker compose logs -f ai-face-service
```

Stop the service:

```bash
docker compose down
```

The `insightface-models` volume caches model weights. `face-data` retains the
legacy SQLite database as a fallback and migration source. With PostgreSQL
enabled, active biometric templates live in PostgreSQL. `docker compose down`
retains named volumes; do not use `docker compose down -v` until any required
SQLite migration is complete.

To use another host port:

```bash
AI_FACE_PORT=8080 docker compose up -d
```

Equivalent manual build and run:

```bash
docker build -t ai-face-service .
docker run --rm \
  -p 8000:8000 \
  --env-file .env \
  -v insightface-models:/home/service/.insightface \
  -v face-data:/service/data \
  ai-face-service
```

The container uses one Uvicorn worker because every worker will load a separate
copy of the InsightFace model into memory. The named model volume preserves
downloaded models between container runs.

## Storage design

Both `PostgreSQLEmbeddingRepository` and `SQLiteEmbeddingRepository` implement
the same repository protocol. PostgreSQL stores an L2-normalized vector as
2,048 bytes in a `BYTEA` column: 512 little-endian float32 values. SQLite uses
the identical binary representation in a BLOB. Neither backend stores the
source face image.

### Shared PostgreSQL with Spring Boot

The Spring Boot service remains the schema owner through Flyway. Migration
`V5__create_face_embeddings.sql` lives in the attendance backend's
`src/main/resources/db/migration/` directory. Start Spring Boot once so Flyway
records and applies it in each environment.

The existing Spring `face_registrations` table remains application metadata.
The new `face_embeddings` table stores the biometric vector in the same
`smart_attendance` database. Both rows use the same Spring student UUID and the
same `embedding_id`, preserving the current REST integration contract.

Spring Boot uses a JDBC URL:

```text
jdbc:postgresql://localhost:5432/smart_attendance
```

The Dockerized Python service uses a Psycopg URL and must address a PostgreSQL
server running on the host as `host.docker.internal`:

```env
FACE_DATABASE_URL=postgresql://smart_attendance:change-me@host.docker.internal:5432/smart_attendance
```

If both services and PostgreSQL share one Compose network, replace
`host.docker.internal` with the PostgreSQL Compose service name. Use separate,
least-privilege database credentials outside local development.

To preserve existing SQLite registrations, first copy the legacy database out
of the running container, then run the one-time migration from the host:

```bash
mkdir -p data
docker cp \
  facerecognizeai-ai-face-service-1:/service/data/faces.db \
  data/faces-from-docker.db

FACE_DATABASE_URL='postgresql://smart_attendance:smart_attendance@localhost:5432/smart_attendance' \
python -m scripts.migrate_sqlite_embeddings_to_postgres \
  --sqlite data/faces-from-docker.db
```

The migration preserves existing embedding IDs so Spring's current
`face_registrations.embedding_id` references remain valid. After migration,
rebuild and recreate the AI service:

```bash
docker compose up -d --build --force-recreate
```

`BYTEA` is appropriate for verification by a known student ID because the
service loads only one vector and computes cosine similarity in Python. Add
`pgvector` later only if the product needs database-side nearest-neighbor face
identification across many students.

## Threshold calibration

Create a CSV containing genuine same-person pairs and impostor
different-person pairs:

```csv
left_image,right_image,is_match
student-001/a.jpg,student-001/b.jpg,true
student-001/a.jpg,student-002/a.jpg,false
```

Image paths are resolved relative to the CSV. Run:

```bash
python -m scripts.tune_threshold \
  --pairs calibration/pairs.csv \
  --thresholds 0.40,0.45,0.50,0.55,0.60 \
  --output reports/thresholds.json
```

The command loads InsightFace once, caches each unique image embedding, and
reports true accepts, false accepts, true rejects, false rejects, FAR, FRR,
accuracy, and half total error rate (HTER) for every threshold. The recommended
threshold minimizes HTER, then favors a smaller FAR/FRR gap and the higher
threshold on a tie.

Build a representative validation dataset and measure false-accept and
false-reject rates across different students, cameras, lighting, pose, and
capture sessions. Do not select the production threshold from one or two
manual comparisons. After reviewing the report, set the selected value in
`.env` as `SIMILARITY_THRESHOLD`.

The quality thresholds are also starting values rather than universal
constants. Tune `MIN_BLUR_SCORE` and `MIN_FACE_SIZE_PIXELS` against valid and
invalid captures from the deployment cameras.

Keep calibration images and generated reports outside source control; the
provided `.gitignore` excludes `calibration/` and `reports/`.

## Performance testing

Start the service, then run the benchmark with a representative face image:

```bash
python -m scripts.benchmark_api \
  --base-url http://127.0.0.1:8000 \
  --image benchmark/face.jpg \
  --endpoints detect,embedding \
  --warmup 3 \
  --requests 20 \
  --concurrency 1 \
  --output reports/benchmark.json
```

To include verification and register the student before measuring:

```bash
python -m scripts.benchmark_api \
  --image benchmark/verification.jpg \
  --endpoints detect,embedding,verify \
  --student-id STU-001 \
  --register-image benchmark/registration.jpg \
  --warmup 3 \
  --requests 20 \
  --concurrency 2 \
  --output reports/benchmark.json
```

The tool reports client-observed and server-reported average, p50, p95, and p99
latency, HTTP status counts, internal pipeline timings, process RSS/VMS memory,
CPU percentages, and thread count.

Run separate cold-start, warmed single-request, and concurrent-request tests.
Do not combine model download/startup time with warmed inference latency.

The combined `insightface_analysis_ms` metric includes detection, alignment,
and recognition because `FaceAnalysis.get()` executes the configured models as
one pipeline. `embedding_postprocess_ms` measures dimension validation and
normalization after InsightFace returns.

## Integration handoff

The independent service is now ready for representative-data calibration and
performance testing before Spring Boot integration. Keep the FastAPI service
as the biometric-processing boundary and have Spring Boot call its REST API.

## Model licensing

The InsightFace library code and its pretrained model weights have different
license terms. The
[InsightFace project](https://github.com/deepinsight/insightface#license)
states that the bundled pretrained model packs, including `buffalo_l`, are for
non-commercial research and asks users to contact InsightFace for
recognition-model licensing. Use a properly licensed model pack before
commercial or institutional production deployment.

Face images and embeddings are biometric data. Define retention,
access-control, encryption, audit, consent, and deletion rules before storing
production embeddings.
