#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "usage: $0 <s3-bucket-name> <s3-bucket-prefix> <local-data-dir>"
    exit 1
fi

S3_BUCKET=$1
S3_PREFIX=$2
DATA_DIR=$3

echo "`date`: Uploading data directory $DATA_DIR to s3://$S3_BUCKET/$S3_PREFIX"
aws s3 cp --recursive $DATA_DIR s3://$S3_BUCKET/$S3_PREFIX | awk 'BEGIN {ORS="="} {if(NR%100==0)print "="}'
echo "Done."

echo "Delete stage directory: $DATA_DIR"
rm -rf $DATA_DIR
echo "Success."