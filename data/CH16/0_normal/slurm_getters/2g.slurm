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


echo "2g"

wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_041.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_041.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_042.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_042.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_043.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_043.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_044.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_044.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_045.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_045.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_047.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_047.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_050.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_050.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_053.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_053.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_054.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_054.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_055.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_055.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_056.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_056.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_057.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_057.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_058.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_058.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_059.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_059.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_060.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_060.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_062.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_062.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_063.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_063.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_067.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_067.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_070.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_070.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_072.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_072.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_076.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_076.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_077.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_077.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_078.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_078.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_080.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_080.tif"

echo "COMPLETE"
