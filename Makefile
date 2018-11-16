build_jj:
	docker-compose -f dockerJJ/docker-compose.yml build
run_jj:
	docker-compose -f dockerJJ/docker-compose.yml up -d
build_e:
	docker-compose -f dockerE/docker-compose.yml build
run_e:
	docker-compose -f dockerE/docker-compose.yml up -d
devel:
	docker exec -it salgan_pytorch bash