
for p in t q g
do
for typ in cc c
do
for c in 2 1 0
do 
sbatch submit_best.sh $p $typ $c 
done
done
done