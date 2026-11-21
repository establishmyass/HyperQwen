FROM docker.io/vllm/vllm-openai:inkling-2-mini@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e

COPY sm89.patch long-context.patch align_sm89.py route_sm89.py test_align.py /tmp/inkling-sm89/
RUN sp=$(python3 -c 'import sysconfig; print(sysconfig.get_path("purelib"))') \
    && patch -d "$sp" -p1 --fuzz=0 < /tmp/inkling-sm89/sm89.patch \
    && patch -d "$sp" -p1 --fuzz=0 < /tmp/inkling-sm89/long-context.patch \
    && install -m 0644 /tmp/inkling-sm89/align_sm89.py /tmp/inkling-sm89/route_sm89.py "$sp/vllm/models/inkling2/nvidia/ops/" \
    && python3 /tmp/inkling-sm89/test_align.py
