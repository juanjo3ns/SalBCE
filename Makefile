build_jj:
	docker-compose -f docker/docker-composeJJ.yml build
run_jj:
	CURRENT_UID=$(id -u):$(id -g)
	docker-compose -f docker/docker-composeJJ.yml up -d
build_e:
	docker-compose -f docker/docker-composeE.yml build
run_e:
	CURRENT_UID=$(id -u):$(id -g)
	docker-compose -f docker/docker-composeE.yml up -d
devel:
	docker exec -it salgan_pytorch bash
