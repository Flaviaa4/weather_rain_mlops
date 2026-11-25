# Flood test summary

| containers | users | requests/s | requests/s vs fewest containers | median ms | p95 ms | failures | failure rate | requests | responses per container |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 300 | 53.31 | 1.0 | 4900.0 | 5800.0 | 0 | 0.00% | 3147 | 1 hosts, 2817 |
| 2 | 300 | 93.61 | 1.76 | 1900.0 | 3900.0 | 0 | 0.00% | 5528 | 2 hosts, 2510 / 2502 |
| 4 | 300 | 90.22 | 1.69 | 1800.0 | 4500.0 | 0 | 0.00% | 5329 | 4 hosts, 1270 / 1223 / 1176 / 1173 |

Breaking point (failures above 1% or p95 above 2000 ms):

- 1 api containers: breaks at 300 users
- 2 api containers: breaks at 300 users
- 4 api containers: breaks at 300 users
