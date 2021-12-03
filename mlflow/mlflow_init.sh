#!/bin/bash

# postgresql must be installed!
# brew install postgresql

# brew services start postgresql
# createdb `whoami`
# createuser -s postgres

echo "Start postgresql"
brew services restart postgresql
sleep 3

echo "Initializing postgresql database"
psql -U postgres -f "db_init.sql"