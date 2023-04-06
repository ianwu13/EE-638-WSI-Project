#!/bin/bash
#SBATCH --account=ajiteshs_1045
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00

# Change to proper directory
if [[ "$PWD" =~ "slurm_getters" ]]
then 
    cd ..
else
    echo 'SLURM JOBS MUST BE RUN FROM "slurm_getters" DIRECTORY FOR THIS PROJECT'
    return -1
fi


echo "1g"

wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_001.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_001.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_002.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_002.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_004.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_004.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_008.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_008.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_010.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_010.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_011.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_011.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_013.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_013.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_016.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_016.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_021.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_021.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_026.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_026.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_027.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_027.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_029.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_029.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_030.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_030.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_033.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_033.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_038.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_038.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_040.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_040.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_046.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_046.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_048.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_048.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_051.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_051.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_052.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_052.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_061.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_061.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_064.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_064.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_065.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_065.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_066.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_066.tif"

echo "COMPLETE"
