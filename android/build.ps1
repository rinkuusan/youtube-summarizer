param(
    [string]$Sdk = "$env:LOCALAPPDATA\Android\Sdk",
    [string]$Jdk = 'C:\Program Files\Android\Android Studio\jbr'
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not $env:APK_KEYSTORE -or -not $env:APK_KEY_PASSWORD) { throw 'Set APK_KEYSTORE and APK_KEY_PASSWORD before building.' }
$env:JAVA_HOME = $Jdk
$env:PATH = "$Jdk\bin;$env:PATH"
$bt = Join-Path $Sdk 'build-tools\35.0.0'
$platform = Join-Path $Sdk 'platforms\android-35\android.jar'
function Run-Checked([string]$File, [string[]]$Arguments) {
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Build failed: $File" }
}
New-Item -ItemType Directory -Force build/classes,build/dex,out | Out-Null
Run-Checked "$bt\aapt2.exe" @('compile','--dir','res','-o','build/resources.zip')
Run-Checked "$bt\aapt2.exe" @('link','-o','build/resources.apk','-I',$platform,'--manifest','AndroidManifest.xml','--min-sdk-version','26','--target-sdk-version','35','-A','assets','build/resources.zip')
$sources = @(Get-ChildItem 'src/jp/tanakabutton/videonotes/*.java' | ForEach-Object FullName)
Run-Checked "$Jdk\bin\javac.exe" (@('-encoding','UTF-8','-source','8','-target','8','-classpath',$platform,'-d','build/classes') + $sources)
$classes = @(Get-ChildItem 'build/classes/jp/tanakabutton/videonotes/*.class' | ForEach-Object FullName)
Run-Checked "$bt\d8.bat" (@('--lib',$platform,'--min-api','26','--output','build/dex') + $classes)
Copy-Item 'build/resources.apk' 'build/unsigned.apk' -Force
Run-Checked "$Jdk\bin\jar.exe" @('uf','build/unsigned.apk','-C','build/dex','classes.dex')
Run-Checked "$bt\zipalign.exe" @('-f','-p','4','build/unsigned.apk','build/aligned.apk')
Run-Checked "$bt\apksigner.bat" @('sign','--ks',$env:APK_KEYSTORE,'--ks-key-alias','video-notes','--ks-pass','env:APK_KEY_PASSWORD','--key-pass','env:APK_KEY_PASSWORD','--out','out/video-notes-1.2.0.apk','build/aligned.apk')
Run-Checked "$bt\apksigner.bat" @('verify','--verbose','out/video-notes-1.2.0.apk')
