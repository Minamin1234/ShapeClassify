docker container stop %1
docker rm %1
docker image build -t %1 .
docker container run ^
    --mount type=bind,src=%~dp0,dst=/home/ShapeClassify/ ^
    -it ^
    -d ^
    --name %1 %1
docker container exec -it %1 /bin/bash