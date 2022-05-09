mkdir traces
for i in 0 1 2
do
    python traceParser.py direct 3 4 ./LinnOSWriterReplayer/py/TrainTraceOutput ./traces/"raw${i}.csv" ./traces/"mldrive${i}.csv" "$i"
done