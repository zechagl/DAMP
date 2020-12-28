python -m src.sketch_relevance_domain.evaluate -g 0 -d recipes -l recipes -o 0.6 -r 0.00001 -b 3 -t 1 -c 60 -p model/sketch_relevance_domain
python -m src.fine_relevance_domain.evaluate -g 0 -d recipes -l recipes -o 0.6 -r 0.00001 -b 3 -t 1 -c 60 -p model/fine_relevance_domain -f 2 -i recipes.hdf5
