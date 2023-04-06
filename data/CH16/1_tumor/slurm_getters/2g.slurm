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

wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_068.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_068.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_069.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_069.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_071.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_071.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_073.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_073.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_074.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_074.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_075.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_075.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_079.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_079.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_082.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_082.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_084.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_084.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_090.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_090.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_092.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_092.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_094.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_094.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_097.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_097.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_099.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_099.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_102.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_102.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_104.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_104.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_105.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_105.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_108.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_108.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_110.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_110.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_113.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_113.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_114.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_114.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_116.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_116.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_117.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_117.tif"
wget https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100439/CAMELYON16/testing/images/test_121.tif
echo "COMPLETED DOWNLOAD FOR CAMELYON16/testing/images/test_121.tif"

echo "COMPLETE"
