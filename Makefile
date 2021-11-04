.PHONY: quality style


check_dirs := api_inference_community tests docker_images


quality:
	black --check $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)

style:
	black $(check_dirs)
	isort $(check_dirs)

test:
	pytest -sv tests/


build:
	docker build -t starlette-app .


dev: build
	docker run -p 8000:80 -e TASK=image-classification -e MODEL_ID=nateraw/coolmodel -v /tmp:/data -t starlette-app
