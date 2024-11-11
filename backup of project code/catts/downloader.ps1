param(
    [string]$s3FolderPath,
    [string]$localDestinationPath
)

aws s3 cp $s3FolderPath $localDestinationPath --recursive


