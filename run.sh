#!E:\Character Motion\CartPole\tianshou_cartpole
echo "Hello world"

#嵌套循环
# for masscart in 0.05 0.1 0.25 0.5 1.0 2.5 4.0 5.5 7.5 10.0
for masscart in 0.1 0.25 1.0 5.0 
do 
for masspole in 0.05 0.1 0.5 1.0 5.0
do
for length in 0.1 0.5 1.0 5.0
do
#echo $masscart $masspole $length
python  ppo_cartpole.py --masscart $masscart --masspole $masspole --length $length
done
done
done


