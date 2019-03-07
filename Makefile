build:
	docker-compose -f docker/docker-compose.yml build
run:
	docker-compose -f docker/docker-compose.yml up -d
build_e:
	docker-compose -f docker/docker-composeE.yml build
run_e:
	docker-compose -f docker/docker-composeE.yml up -d
devel:
	docker exec -it salgan_pytorch bash
down:
	docker-compose -f docker/docker-compose.yml down -v
down_e:
	docker-compose -f docker/docker-composeE.yml down -v
