include env.sh

run:
	python $(ENTRYPOINT_PATH)

docker_reset:
	docker stop $(DOCKER_CONTAINER_NAME) && docker rm $(DOCKER_CONTAINER_NAME) && docker rmi $(DOCKER_IMAGE_NAME)

docker_run:
	@echo "Deleting container and image"
	-make docker_reset
	wait

	@echo "Building image: $(DOCKER_IMAGE_NAME), HDA_INTEGRATION_PATH_IN_CONTAINER=$(HDA_INTEGRATION_PATH_IN_CONTAINER)"
	docker build --build-arg HDA_INTEGRATION_PATH_IN_CONTAINER=$(HDA_INTEGRATION_PATH_IN_CONTAINER) -t $(DOCKER_IMAGE_NAME) .

	@echo "Starting container: $(DOCKER_CONTAINER_NAME)"
	docker run --name $(DOCKER_CONTAINER_NAME) -d -i -t\
		-v $(PWD):$(HDA_INTEGRATION_PATH_IN_CONTAINER)\
		-p 8088:8088 $(DOCKER_IMAGE_NAME)\

	@echo "Container '$(DOCKER_CONTAINER_NAME)' started"

docker_enter_container:
	docker exec -it $(DOCKER_CONTAINER_NAME) bash