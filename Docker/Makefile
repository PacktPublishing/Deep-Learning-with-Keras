help:
	@cat Makefile

DATA?="${HOME}/Data"
GPU?=0
DOCKER_FILE=Dockerfile
DOCKER_FILE_GPU=Dockerfile-gpu
DOCKER_FILE_WAVNET=Dockerfile-wavnet
DOCKER=docker
DOCKER_GPU=GPU=$(GPU) nvidia-docker
BACKEND=tensorflow
TEST=tests/
SRC=$(shell dirname `pwd`)
CUDA=/usr/local/cuda
LD_LIBRARY=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64

# CPU environment

build:
	$(DOCKER) build -t deep-learning-with-keras --build-arg python_version=3.5 -f $(DOCKER_FILE) .

bash: build
	$(DOCKER) run -p 8889:8889 -p 6007:6007 -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $(SRC):/src -v $(DATA):/data deep-learning-with-keras bash

ipython: build
	$(DOCKER) run -p 8888:8888 -p 6006:6006 -it -v $(SRC):/src -v $(DATA):/data deep-learning-with-keras ipython

notebook: build
	$(DOCKER) run -p 8888:8888 -p 6006:6006 -it -v $(SRC):/src -v $(DATA):/data deep-learning-with-keras

test: build
	$(DOCKER) run -it -v $(SRC):/src -v $(DATA):/data deep-learning-with-keras pytest $(TEST)

# GPU environment

build-gpu:
	$(DOCKER_GPU) build -t deep-learning-with-keras-gpu --build-arg python_version=3.5 -f $(DOCKER_FILE_GPU) .

bash-gpu: build-gpu
	$(DOCKER_GPU) run -p 8888:8888 -p 6006:6006 -it -v $(SRC):/src -v $(DATA):/data --env CUDA_HOME=${CUDA} \
	                                                  -e DISPLAY=$DISPLAY \
	                                                  -v /tmp/.X11-unix:/tmp/.X11-unix \
	                                                  --env LD_LIBRARY_PATH=${LD_LIBRARY} \
	                                                  deep-learning-with-keras-gpu bash
ipython-gpu: build-gpu
	$(DOCKER_GPU) run -p 8888:8888 -p 6006:6006  -it -v $(SRC):/src -v $(DATA):/data --env CUDA_HOME=${CUDA} \
	                                                  --env LD_LIBRARY_PATH=${LD_LIBRARY} \
	                                                  deep-learning-with-keras-gpu ipython

notebook-gpu: build-gpu
	$(DOCKER_GPU) run -p 8888:8888 -p 6006:6006 -it -v $(SRC):/src -v $(DATA):/data --env CUDA_HOME=${CUDA} \
	                                                  --env LD_LIBRARY_PATH=${LD_LIBRARY} \
	                                                  deep-learning-with-keras-gpu

test-gpu: build-gpu
	$(DOCKER_GPU) run -it -v $(SRC):/src -v $(DATA):/data --env CUDA_HOME=${CUDA} \
	                                                  --env LD_LIBRARY_PATH=${LD_LIBRARY} \
	                                                  deep-learning-with-keras-gpu pytest $(TEST)

# CPU environment Wave net

build-wavenet:
	$(DOCKER_GPU) build -t deep-learning-with-keras-wavenet --build-arg python_version=3.5 -f $(DOCKER_FILE_WAVNET) .

bash-wavenet: build-wavenet
	$(DOCKER_GPU) run -it -v $(SRC):/src --env CUDA_HOME=${CUDA} \
	                                                --env LD_LIBRARY_PATH=${LD_LIBRARY} \
	                                                deep-learning-with-keras-wavenet bash

ipython-wavenet: build-wavenet
	$(DOCKER_GPU) run -p 8888:8888 -p 6006:6006 -it -v $(SRC):/src --env CUDA_HOME=${CUDA} \
	                                                --env LD_LIBRARY_PATH=${LD_LIBRARY} \
	                                                deep-learning-with-keras-wavenet ipython

notebook-wavenet: build-wavenet
	$(DOCKER_GPU) run -p 8888:8888 -p 6006:6006 -it -v $(SRC):/src --env CUDA_HOME=${CUDA} \
	                                                --env LD_LIBRARY_PATH=${LD_LIBRARY} \
	                                                deep-learning-with-keras-wavenet

test-wavenet: build-wavenet
	$(DOCKER_GPU) run -it -v $(SRC):/src --env CUDA_HOME=${CUDA} \
	                                     --env LD_LIBRARY_PATH=${LD_LIBRARY} \
	                                     deep-learning-with-keras-wavenet pytest $(TEST)