FROM python:3.12

RUN pip install --upgrade pip

WORKDIR /spare_score
COPY ./ ./
RUN pip install -e .

# Download all registered default-version models into the HF cache at build time
# so the image works without internet access at runtime.
RUN python -c "from NiChart_SPARE.task_registry import download_all_default_models; download_all_default_models()"

ENTRYPOINT ["NiChart_SPARE"]
