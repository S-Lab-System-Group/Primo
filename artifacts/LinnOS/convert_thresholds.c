#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "xxx/w_sdb.h" // You need to get w_sdb/c/d.h first. We obtain them manually.

//./convert_thresholds > ./intRecord.txt
void main()
{
    int len = sizeof(thresholds) / sizeof(thresholds[0]);
    int i, intTh[len];
    printf("int ithresholds[%d] = {", len);
    for (i = 0; i < len; i++)
    {
        if (thresholds[i] > 0)
        {
            intTh[i] = (int)(thresholds[i] + 0.5);
        }
        else
        {
            intTh[i] = (int)(thresholds[i]);
        }
        if (i < len - 1)
        {
            printf("%d,", intTh[i]);
        }
        else
        {
            printf("%d", intTh[i]);
        }
    }
    printf("};");
}