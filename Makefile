build:
	docker-compose -f docker/docker-compose.yml build
run:
	docker-compose -f docker/docker-compose.yml up -d
devel:
	docker exec -it salgan_pytorch bash