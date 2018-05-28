for i in `seq 0 30`;
do
    python3 test$i.py > output$i.txt &
done
