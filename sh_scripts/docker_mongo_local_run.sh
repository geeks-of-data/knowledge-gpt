docker run  -d -p 27017:27017 --name container_name \
      -e MONGO_INITDB_ROOT_USERNAME=user \
      -e MONGO_INITDB_ROOT_PASSWORD=secret_password \
      mongo