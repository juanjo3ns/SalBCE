build_jj:
	docker-compose -f docker/docker-composeJJ.yml build
run_jj:
	docker-compose -f docker/docker-composeJJ.yml up -d
build_e:
	docker-compose -f docker/docker-composeE.yml build
run_e:
	docker-compose -f docker/docker-composeE.yml up -d
devel:
	docker exec -it salgan_pytorch bash