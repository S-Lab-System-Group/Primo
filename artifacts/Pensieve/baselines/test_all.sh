# echo "Trace: HSDPA  Algorithm: Primo"
# python primo.py -hsdpa

# echo "Trace: HSDPA  Algorithm: Pensieve"
# python rl_no_training.py -hsdpa

# cd metis
# echo "Trace: HSDPA  Algorithm: Metis"
# python metis_test.py -hsdpa
# cd ..

# echo "Trace: HSDPA  Algorithm: BB"
# python bb.py -hsdpa

echo "Trace: HSDPA  Algorithm: MPC"
python mpc.py -hsdpa


# echo "Trace: FCC  Algorithm: Primo"
# python primo.py -fcc

# echo "Trace: FCC  Algorithm: Pensieve"
# python rl_no_training.py -fcc

# cd metis
# echo "Trace: FCC  Algorithm: Metis"
# python metis_test.py -fcc
# cd ..

# echo "Trace: FCC  Algorithm: BB"
# python bb.py -fcc

echo "Trace: FCC  Algorithm: MPC"
python mpc.py -fcc

