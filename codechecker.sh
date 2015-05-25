#!/bin/bash

UNAME=$(uname)
UBUNTU_CODENAME=$(lsb_release -sc)
BASEDIR="/tmp/codechecker_install_logs"
mkdir -p $BASEDIR
exec 3<> "$BASEDIR/install.$$.log"
BASH_XTRACEFD=3
set -x

function display_error() {
    echo -e "\n\033[31;47;1m$*\033[0m"
    exit 1
}

function display_ok() {
    echo -e "\033[32;1mOK\033[0m"
}

function display_fetched() {
    echo -e "\033[32;1mFetched\033[0m"
}

function add_apt_repo() {
    echo -n "Adding repo $1..."
    sudo add-apt-repository -y $1 1>&3 2>&3
    if [[ $? -ne 0 ]]
    then
        display_error "Error in adding repository $1"
        exit 1
    fi
    display_ok
}

function add_apt_public_key() {
    echo -n "Adding public key from $1"

    KEY_LOC="/tmp/$(basename $1).$$"
    wget_link $1 $KEY_LOC
    sudo apt-key add $KEY_LOC

    if [[ $? -ne 0 ]]
    then
        display_error "Error in adding public key from $1"
        exit 1
    fi
    display_ok
}

function fastwget_link() {
    #Example fastwget_link link folderpath filename
    echo -n "Fetching $1..."
    aria2c --help > /dev/null 2>&1
    if [[ $? -ne 0 ]]
    then
        wget_link $1 $2/$3
    else
        sudo aria2c -d $2 -o $3 $1 1>&3 2>&3
    fi
    if [[ $? -ne 0 ]]
    then
        display_error "Wget failed for $1 copied at $2"
        exit 1
    fi
    display_fetched
}

function wget_link() {
    echo -n "Fetching $1..."
    sudo wget -O $2 $1 1>&3 2>&3
    if [[ $? -ne 0 ]]
    then
        display_error "Wget failed for $1 copied at $2"
        exit 1
    fi
    display_fetched
}

function setup_aptfast() {
    echo -n "Installing apt-fast..."
    fastwget_link https://raw.githubusercontent.com/ilikenwf/apt-fast/master/apt-fast.conf /etc apt-fast.conf
    fastwget_link https://raw.githubusercontent.com/ilikenwf/apt-fast/master/apt-fast /usr/bin apt-fast
    sudo chmod 755 /usr/bin/apt-fast
}

function aptget_upgrade() {
    echo -n "Upgrading...."
    if [ "$UNAME" == "Linux" ]; then
        sudo apt-fast update 1>&3 2>&3
        sudo apt-fast upgrade -y 1>&3 2>&3
    else
        brew update 1>&3 2>&3
        brew upgrade 1>&3 2>&3
    fi
    display_ok
}

function aptget_install() {
    echo -n "Installing $1..."
    if [ "$UNAME" == "Linux" ]; then
        sudo apt-get --assume-yes --verbose-versions install $1 1>&3 2>&3
    else
        brew install $1 1>&3 2>&3
    fi
    if [[ $? -ne 0 ]]
        then
        display_error "Error in installing $1"
    fi
    display_ok
}

function aptfast_install() {
    echo -n "Installing $1..."
    if [ "$UNAME" == "Linux" ]; then
        sudo apt-fast -y --force-yes -V install $1 1>&3 2>&3
    else
        brew install $1 1>&3 2>&3
    fi
    if [[ $? -ne 0 ]]
        then
        display_error "Error in installing $1"
    fi
    display_ok
}

function install_osx_essentials() {
    echo -n "Installing OS-X essentials"
    # install homebrew

    ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)" 1>&3 2>&3
    display_ok
}

function install_haskell() {
    echo -n "Installing LTS haskell..."
    GHC_VERSION=7.8.4
    aptfast_install ghc-$GHC_VERSION
    aptfast_install ghc-$GHC_VERSION-dyn
    # Lets have a current folder pointing to latest to make life easier
    ln -s /opt/ghc/$GHC_VERSION /opt/ghc/current

    aptfast_install cabal-install-1.16
    cabal-1.16 update 1>&3 2>&3
    if [[ $? -ne 0 ]]
    then
        display_error "Failed to update cabal-1.16"
        exit 1
    fi

    STACKAGE_LINK="http://www.stackage.org/lts/cabal.config"
    STACKAGE_CABAL_CONFIG="/tmp/cabal.config.$$"
    wget_link $STACKAGE_LINK $STACKAGE_CABAL_CONFIG                                 #Download config file from stackage server
    REMOTE_REPO=`awk '$0 ~ /'remote-repo:'/ {print $3}' $STACKAGE_CABAL_CONFIG`     #Parse remote repository

    CABAL_CONFIG_TMP="/tmp/cabal_config.$$"
    CABAL_CONFIG="$HOME/.cabal/config"
    awk -v repo=$REMOTE_REPO '{ if ($1 == "remote-repo:") {$2 = repo}; print }' $CABAL_CONFIG > $CABAL_CONFIG_TMP   # Update ~/.cabal/config file.
    cp $CABAL_CONFIG "$CABAL_CONFIG.orig"       #Copy the original cabal config file.
    mv $CABAL_CONFIG_TMP $CABAL_CONFIG

    cabal-1.16 update  1>&3 2>&3
    if [[ $? -ne 0 ]]
    then
        display_error "Failed to update cabal-1.16"
        exit 1
    fi
    cabal-1.16 install --global cabal-install 1>&3 2>&3     #Update cabal to stackage compliance server
    if [[ $? -ne 0 ]]
    then
        cabal install --global --reinstall --force-reinstalls cabal-install 1>&3 2>&3
        if [[ $? -ne 0 ]]
        then
            display_error "[Haskell] Failed to upgrade cabal-install"
            exit 1
        fi
    fi
    display_ok
}

function install_haskell_libs() {
    echo -n "Installing haskell libraries..."
    cabal update 1>&3 2>&3
    if [[ $? -ne 0 ]]
    then
        display_error "Failed to update cabal."
        exit 1
    fi

    PKG_WITH_EXECUTABLES="alex happy"
    for PKG in $PKG_WITH_EXECUTABLES; do
        PKG_LOC=`which $PKG`
        if [ -z "$PKG_LOC" ]
        then
            cabal install --global $PKG 1>&3 2>&3
            if [[ $? -ne 0 ]]
            then
                cabal install --global --reinstall --force-reinstalls $PKG 1>&3 2>&3
                if [[ $? -ne 0 ]]
                then
                    display_error "[Haskell] Failed to install $PKG"
                    exit 1
                fi
            fi
        fi
    done

    HSK_PKG="base base16-bytestring base64-bytestring base-compat base-prelude"
    HSK_PKG="$HSK_PKG random text mtl stm"
    HSK_PKG="$HSK_PKG vector hashmap aeson logict pipes mwc-random hashtables"
    HSK_PKG="$HSK_PKG lens lens-aeson lens-family-th"
    HSK_PKG="$HSK_PKG pqueue bytestring aeson-pretty array arrow-list"
    HSK_PKG="$HSK_PKG regex-applicative regex-base regex-compat regex-pcre-builtin"
    HSK_PKG="$HSK_PKG regex-posix regexpr regex-tdfa"
    HSK_PKG="$HSK_PKG hmatrix multimap generic-aeson"
    HSK_PKG="$HSK_PKG attoparsec attoparsec-conduit attoparsec-enumerator attoparsec-expr"
    HSK_PKG="$HSK_PKG base-unicode-symbols basic-prelude bifunctors"
    HSK_PKG="$HSK_PKG comonad deepseq dlist either matrix MemoTrie pretty-show threads"

    # NOTE: It is guaranteed to be fail on few of the packages. In that case forcefully
    # reinstall the package which `ld` is not able to link and
    # restart from this function excluding CORE_PKG (if already installed).
    for PKG in $HSK_PKG; do
        cabal install --global $PKG 1>&3 2>&3
        if [[ $? -ne 0 ]]
        then
            cabal install --global --reinstall --force-reinstalls $PKG 1>&3 2>&3
            if [[ $? -ne 0 ]]
            then
                display_error "[Haskell] Failed to install $PKG"
                exit 1
            fi
        fi
    done

    display_ok
}

function install_ocaml_libs() {
    echo -n "Installing ocaml libraries..."

    sudo opam init --yes 1>&3 2>&3
    sudo opam switch 4.02.1 --yes 1>&3 2>&3
    sudo chown -R ubuntu:ubuntu /home/ubuntu/.opam
    eval `opam config env`
    sudo opam update --yes 1>&3 2>&3
    sudo opam upgrade --yes 1>&3 2>&3

    sudo opam install --yes core async core_extended ocamlfind 1>&3 2>&3

    if [[ $? -ne 0 ]]
    then
        display_error "Ocaml libs installation failed"
        exit 1
    fi
    sudo cp ~/files/.ocamlinit ~/.ocamlinit
    display_ok
}

function install_erlang() {
    echo -n "Installing erlang..."

    ERL_PKG_LINK="http://packages.erlang-solutions.com/site/esl/esl-erlang/FLAVOUR_1_esl/esl-erlang_17.4-2~ubuntu~trusty_amd64.deb"
    ERL_PKG_LOCAL="/tmp/erlang.$$.deb"

    wget_link $ERL_PKG_LINK $ERL_PKG_LOCAL
    sudo dpkg -i $ERL_PKG_LOCAL

    if [[ $? -ne 0 ]]
    then
        display_error "Erlang installation failed"
        exit 1
    fi
    display_ok
}

function install_boost_from_source() {
    echo -n "Installing boost..."
    [ -d /opt/boost ] && return
    fastwget_link http://sourceforge.net/projects/boost/files/boost/1.56.0/boost_1_56_0.tar.gz/download /tmp boost.tar.gz
    cd /tmp/
    tar -zxvf boost.tar.gz 1>&3 2>&3
    sudo chmod 777 boost_*
    cd boost_*
    sudo ./bootstrap.sh --exec-prefix=/opt/boost/ --libdir=/opt/boost/lib/ --includedir=/opt/boost/include/ 1>&3 2>&3
    sudo ./b2 install 1>&3 2>&3
    display_ok
}

function install_sunjava7() {
    echo -n "Installing sun java7..."
    [ -d /usr/lib/jvm/java-7-sun ] && return
    sudo wget --no-check-certificate --no-cookies -O "/tmp/sunjava.tar.gz" --header "Cookie: gpw_e24=http%3A%2F%2Fwww.oracle.com; oraclelicense=accept-securebackup-cookie;" "http://download.oracle.com/otn-pub/java/jdk/7u55-b13/jdk-7u55-linux-x64.tar.gz" 1>&3 2>&3
    sudo tar -zxvf "/tmp/sunjava.tar.gz" 1>&3 2>&3
    sudo mv -v jdk1.7.0_* "/usr/lib/jvm/java-7-sun/" 1>&3 2>&3
    display_ok
}

function install_sunjava8() {
    echo -n "Installing sun java8..."
    [ -d /usr/lib/jvm/java-8-sun ] && return
    sudo wget --no-check-certificate --no-cookies -O "/tmp/sunjava8.tar.gz" --header "Cookie: gpw_e24=http%3A%2F%2Fwww.oracle.com; oraclelicense=accept-securebackup-cookie;" "http://download.oracle.com/otn-pub/java/jdk/8u5-b13/jdk-8u5-linux-x64.tar.gz" 1>&3 2>&3
    sudo tar -zxvf "/tmp/sunjava8.tar.gz" 1>&3 2>&3
    sudo mv -v jdk1.8.* "/usr/lib/jvm/java-8-sun/" 1>&3 2>&3
    display_ok
}

function install_dart() {
    echo -n "Installing dart..."
    [ -d /usr/local/dart ] && return
    fastwget_link "http://storage.googleapis.com/dart-archive/channels/stable/release/latest/sdk/dartsdk-linux-x64-release.zip" /tmp dart.zip
    cd /tmp/
    unzip -o -d "/tmp/" "/tmp/dart.zip" 1>&3 2>&3
    chmod -R 755 dart-sdk/
    sudo mv -v dart-sdk /usr/local/dart
    display_ok
}

function install_node() {
    echo -n "Installing nodejs..."
    [ -d /usr/local/node ] && return
    fastwget_link "http://nodejs.org/dist/v0.10.28/node-v0.10.28-linux-x64.tar.gz" "/tmp" "node.tar.gz"
    cd /tmp
    tar -zxvf node.tar.gz 1>&3 2>&3
    sudo mv node-* /usr/local/node
    display_ok
}

function install_dmd() {
    echo -n "Installing dmd..."
    dpkg -l dmd 1>&3 2>&3
    [ $? = 0 ] && return # Returns 0 if the package exist else 127
    sudo apt-get -f install --assume-yes gcc-multilib 1>&3 2>&3
    fastwget_link "http://ftp.digitalmars.com/dmd_2.067.0~b1-0_amd64.deb" "/tmp" "dmd.deb"
    sudo dpkg -i /tmp/dmd.deb
    display_ok
}

function install_scala() {
    echo -n "Installing scala..."
    [ -d /usr/local/scala ] && return
    fastwget_link "http://www.scala-lang.org/files/archive/scala-2.11.5.tgz" "/tmp" "scala.tar.gz"
    cd /tmp
    tar -zxvf scala.tar.gz
    sudo mv scala-* /usr/local/scala
    display_ok
}

function install_lolcode() {
    echo -n "Installing lolcode..."
    [ -d /usr/local/lolcode ] && return
    fastwget_link "https://github.com/justinmeza/lci/archive/master.zip" "/tmp" "lolcode.zip"
    cd /tmp
    unzip -uo lolcode.zip 1>&3 2>&3
    chmod -R 755 lci*
    cd lci*
    sudo ./install.py --prefix="/usr/local/lolcode" 1>&3 2>&3
    display_ok
}

function install_sbcl() {
    echo -n "Installing SBCL..."
    [ -d /usr/local/lib/sbcl ] && return
    fastwget_link "http://sourceforge.net/projects/sbcl/files/sbcl/1.2.3/sbcl-1.2.3-x86-64-linux-binary.tar.bz2/download" "/tmp" "sbcl.tar.bz2"
    cd /tmp
    tar -jxvf sbcl.tar.bz2
    sudo chmod -R 755 sbcl-*
    cd sbcl-*
    sudo INSTALL_ROOT=/usr/local sh install.sh 1>&3 2>&3
    if [[ $? -ne 0 ]]
    then
        echo "SBCL install failed."
    fi

    display_ok
}

function install_whitespace() {
    echo -n "Installing whitespace..."
    [ -d /usr/local/whitespace ] && return
    fastwget_link "http://compsoc.dur.ac.uk/whitespace/downloads/wspace" "/tmp" "wspace"
    sudo mkdir /usr/local/whitespace
    sudo cp /tmp/wspace /usr/local/whitespace
    sudo chmod -R 755 /usr/local/whitespace
    display_ok
}

function install_oracle_client() {
    echo -n "Installing oracle client..."
    fastwget_link "https://static.interviewstreet.com/codechecker/oracle-instantclient12.1-basic-12.1.0.1.0-1.x86_64.rpm" "/tmp" "oracle-instantclient12.1-basic-12.1.0.1.0-1.x86_64.rpm"
    sudo alien --install /tmp/oracle-instantclient12.1-basic-12.1.0.1.0-1.x86_64.rpm 1>&3 2>&3

    fastwget_link "https://static.interviewstreet.com/codechecker/oracle-instantclient12.1-sqlplus-12.1.0.1.0-1.x86_64.rpm" "/tmp" "oracle-instantclient12.1-sqlplus-12.1.0.1.0-1.x86_64.rpm"
    sudo alien --install /tmp/oracle-instantclient12.1-sqlplus-12.1.0.1.0-1.x86_64.rpm 1>&3 2>&3
    display_ok
}

function install_db2_client() {
    echo -n "Installing db2 client..."
    fastwget_link "https://s3.amazonaws.com/codechecker-install-essentials/DB2Client.zip" "/tmp" "DB2Client.zip"

    cd /tmp
    unzip DB2Client.zip
    ./DB2Client/db2_install > /dev/null 2>&1

    display_ok
}

function pip_upgrade() {
    echo -n "Upgrading pip..."
    sudo pip install --upgrade pip 1>&3 2>&3
    if [[ $? -ne 0 ]]
    then
        echo "Pip upgrade failed. Proceeding without upgrading pip."
    fi
    display_ok
}

function source_activate()
{
    echo -n "Entering virtualenv of $1."
    source $1 1>&3 2>&3
    if [[ $? -ne 0 ]]
    then
        display_error "Activating virtualenv failed for $1. Install these manually."
        exit 1
    fi
    display_ok
}

function make_virtualenv()
{
    echo -n "Installing virtualenv..."
    sudo virtualenv -p $1 $2 1>&3 2>&3
    if [[ $? -ne 0 ]]
    then
        display_error "Virtualenv installation failed for $1 at $2"
        exit 1
    fi
    display_ok
}

function pip_install()
{
    echo -n "Running pip install of $2..."
    if [ $1 = "python" ]
    then
        language="python"
        version="2.*"
    else
        language="python3"
        version="3.*"
    fi

    package_installation_name=$2
    if [ $2 = "scikit-learn" ] #scikit-learn package is saved as folder name sklearn
    then
        package_installation_name=sklearn
    elif [ $2 = "pyyaml" ] #pyyaml package is saved as folder name yaml
    then
        package_installation_name=yaml
    fi

    [ -d /var/ml/$language/lib/python$version/site-packages/$package_installation_name ] && return

    if [ $2 = "scikit-learn" ]
    then
        # Hack: Build development version directly from git. Source: https://github.com/scikit-learn/scikit-learn/issues/2988
        sudo /var/ml/$language/bin/pip install -t /var/ml/$language/lib/python$version/site-packages/ git+https://github.com/scikit-learn/scikit-learn.git 1>&3 2>&3
    elif [ $1 = "python3" ] && [ $2 = "nltk" ]
    then
        sudo /var/ml/$language/bin/pip install -t /var/ml/$language/lib/python$version/site-packages/ git+https://github.com/nltk/nltk.git 1>&3 2>&3
    else
        sudo /var/ml/$language/bin/pip install -t /var/ml/$language/lib/python$version/site-packages/ $2 1>&3 2>&3
    fi

    if [[ $? -ne 0 ]]
    then
        display_error "Installation of $2 inside virtual env of $language failed. Please check the log file"
        exit 1
    fi
    display_ok
}

function install_ml_python_libs() {

    echo -n "Installing ML libraries for python..."
    export LAPACK=/usr/lib/liblapack.so
    export ATLAS=/usr/lib/libatlas.so
    export BLAS=/usr/lib/libblas.so

    # Install virtualenv at /var/ml
    make_virtualenv /usr/bin/python /var/ml/python
    pip_upgrade
    source_activate /var/ml/python/bin/activate

    # Now install scipy, numpy, and other stuff to this virtualenv
    pip_install python numpy
    pip_install python scipy
    pip_install python scikit-learn
    pip_install python nltk
    pip_install python pandas

    deactivate
    display_ok
}

function install_ml_python3_libs() {

    echo -n "Installing ML libraries for python3..."
    make_virtualenv /usr/bin/python3 /var/ml/python3
    pip_upgrade
    source_activate /var/ml/python3/bin/activate
    pip_install python3 numpy
    pip_install python3 scipy
    pip_install python3 scikit-learn
    pip_install python3 pyyaml
    pip_install python3 pandas
    pip_install python3 nltk

    deactivate
    display_ok
}

function install_ml_java_lib() {
    echo -n "Installing $2..."
    SRC="/var/ml/java"
    fastwget_link $1 "/tmp" "$2.zip"
    unzip -o -d "/tmp/" "/tmp/$2.zip" 1>&3 2>&3
    cd "/tmp"
    [ -d $2 ] && cd $2
    sudo mv -v *.jar $SRC 1>&3 2>&3
    display_ok
}

function install_ml_java_libs() {
    echo -n "Installing ML libraries for java..."
    SRC="/var/ml/java"
    sudo mkdir -p $SRC

    install_ml_java_lib "https://s3.amazonaws.com/codechecker-install-essentials/stanford-corenlp-full-2013-06-20.zip" "stanford-corenlp-full-2013-06-20"
    install_ml_java_lib "https://s3.amazonaws.com/codechecker-install-essentials/stanford-classifier-2013-06-20.zip" "stanford-classifier-2013-06-20"
    install_ml_java_lib "http://downloads.sourceforge.net/project/java-ml/java-ml/javaml-0.1.7.zip" "javaml-0.1.7"
    install_ml_java_lib "http://prdownloads.sourceforge.net/weka/weka-3-6-10.zip" "weka-3-6-10"
    install_ml_java_lib "http://downloads.sourceforge.net/project/weka/weka-packages/LibSVM1.0.5.zip" "LibSVM1.0.5"
    install_ml_java_lib "http://downloads.sourceforge.net/project/ajt/ajt/ajt-2.9.zip" "ajt-2.9"

    fastwget_link "http://math.nist.gov/javanumerics/jama/Jama-1.0.3.jar" "$SRC" "Jama-1.0.3.jar"
    display_ok

    cd $BASEDIR
}

function install_ml_cpp_lib() {
    echo -n "Installing $2..."
    SRC="/var/ml/cpp"
    sudo mkdir -p "$SRC" "$SRC/include" "$SRC/lib"
    cd $SRC
    fastwget_link $1 "/tmp" "$2.zip"
    unzip -o -d "/tmp/" "/tmp/$2.zip" 1>&3 2>&3
    cd /tmp/$2-*
    sudo make lib 1>&3 2>&3
    sudo mv $2.so.* /usr/lib/
    sudo mv *.h $SRC/include/
    display_ok
}

function install_ml_cpp_libs() {
    echo -n "Installing ML libraries for cpp..."
    SRC="/var/ml/cpp"
    install_ml_cpp_lib 'http://www.csie.ntu.edu.tw/~cjlin/cgi-bin/libsvm.cgi?+http://www.csie.ntu.edu.tw/~cjlin/libsvm+zip' 'libsvm'

    install_ml_cpp_lib 'http://www.csie.ntu.edu.tw/~cjlin/liblinear/oldfiles/liblinear-1.94.zip' 'liblinear'

    cd "/usr/lib/"
    sudo ln -sfn libsvm.so.2 libsvm.so
    sudo ln -sfn liblinear.so.1 liblinear.so
    sudo ldconfig
    display_ok
    cd $BASEDIR
}

function install_scala_twitter_lib() {
    echo -n "Installing $1..."
    TWITTER_BASE="/var/ml/java/twitter/"
    [ ! -d $TWITTER_BASE ] && sudo mkdir $TWITTER_BASE

    LIB_BASE="$TWITTER_BASE/$3/"
    [ ! -d $LIB_BASE ] && sudo mkdir $LIB_BASE

    [ -f $LIB_BASE/$2 ] && return

    fastwget_link $1 "/tmp" "$2"
    sudo mv "/tmp/$2" $LIB_BASE 1>&3 2>&3

    display_ok
}

function install_scala_twitter_libs() {
    echo -n "Installing twitter libs for scala..."

    install_scala_twitter_lib 'http://central.maven.org/maven2/com/twitter/algebird-core_2.10/0.6.0/algebird-core_2.10-0.6.0.jar' 'algebird-core.jar' 'algebird'
    install_scala_twitter_lib 'http://central.maven.org/maven2/com/twitter/algebird-util_2.10/0.6.0/algebird-util_2.10-0.6.0.jar' 'algebird-util.jar' 'algebird'
    install_scala_twitter_lib 'http://central.maven.org/maven2/com/twitter/algebird-bijection_2.10/0.6.0/algebird-bijection_2.10-0.6.0.jar' 'algebird-bijection.jar' 'algebird'
    install_scala_twitter_lib 'http://central.maven.org/maven2/com/twitter/algebird-test_2.10/0.6.0/algebird-test_2.10-0.6.0.jar' 'algebird-test.jar' 'algebird'
    install_scala_twitter_lib 'http://central.maven.org/maven2/com/twitter/algebird_2.10/0.6.0/algebird_2.10-0.6.0.jar' 'algebird.jar' 'algebird'

    #install_scala_twitter_lib 'http://central.maven.org/maven2/com/twitter/scalding-repl_2.10/0.10.0/scalding-repl_2.10-0.10.0.jar' 'scalding-repl.jar' 'scalding'
    #install_scala_twitter_lib 'http://central.maven.org/maven2/com/twitter/scalding-json_2.10/0.10.0/scalding-json_2.10-0.10.0.jar' 'scalding-json.jar' 'scalding'
    #install_scala_twitter_lib 'http://central.maven.org/maven2/com/twitter/scalding-jdbc_2.10/0.10.0/scalding-jdbc_2.10-0.10.0.jar' 'scalding-jdbc.jar' 'scalding'
    #install_scala_twitter_lib 'http://central.maven.org/maven2/com/twitter/scalding-avro_2.10/0.10.0/scalding-avro_2.10-0.10.0.jar' 'scalding-avro.jar' 'scalding'
    #install_scala_twitter_lib 'http://central.maven.org/maven2/com/twitter/scalding-core_2.10/0.10.0/scalding-core_2.10-0.10.0.jar' 'scalding-core.jar' 'scalding'
    #install_scala_twitter_lib 'http://central.maven.org/maven2/com/twitter/scalding-date_2.10/0.10.0/scalding-date_2.10-0.10.0.jar' 'scalding-date.jar' 'scalding'
    #install_scala_twitter_lib 'http://central.maven.org/maven2/com/twitter/scalding-args_2.10/0.10.0/scalding-args_2.10-0.10.0.jar' 'scalding-args.jar' 'scalding'
    #install_scala_twitter_lib 'http://central.maven.org/maven2/com/twitter/scalding-commons_2.10/0.10.0/scalding-commons_2.10-0.10.0.jar' 'scalding-commons.jar' 'scalding'
    #install_scala_twitter_lib 'search.maven.org/remotecontent?filepath=com/twitter/scalding-parquet_2.10/0.11.0/scalding-parquet_2.10-0.11.0.jar' 'scalding-parquet.jar' 'scalding'

    display_ok
}

function install_java_lib() {
    echo -n "Installing $1"
    JAVA_LIBS_BASE="/usr/share/java/"
    [ -f $JAVA_LIBS_BASE/$2 ] && return

    wget_link $1 "/tmp/$2"
    sudo mv "/tmp/$2" $JAVA_LIBS_BASE 1>&3 2>&3
}

function install_miscellaneous_java_libs() {
    echo -n "Installing miscellaneous java libs..."
    install_java_lib 'http://search.maven.org/remotecontent?filepath=org/testng/testng/6.8.8/testng-6.8.8.jar' 'testng.jar'
    install_java_lib 'http://json-simple.googlecode.com/files/json-simple-1.1.1.jar' 'json-simple-1.1.1.jar'
    install_java_lib 'http://home.ccil.org/~cowan/tagsoup/tagsoup-1.2.1.jar' 'tagsoup.jar'

    display_ok
}

function install_clojure() {
    echo -n "Installing clojure..."
    install_java_lib 'http://central.maven.org/maven2/org/clojure/clojure/1.6.0/clojure-1.6.0.jar' 'clojure-1.6.0.jar'

    display_ok
}

function install_junixsocket() {
    echo -n "Installing junixsocket..."

    JAVA_LIB="junixsocket.jar"
    NATIVE_LIB="libjunixsocket-linux-1.5-amd64.so"

    [[ -f /usr/share/java/$JAVA_LIB && -f /usr/lib/$NATIVE_LIB ]] && return

    wget_link "http://junixsocket.googlecode.com/files/junixsocket-1.3-bin.tar.bz2" "/tmp/junixsocket.tar.bz2"
    cd /tmp
    tar -jxvf "junixsocket.tar.bz2" 1>&3 2>&3
    cd junixsocket*
    sudo cp dist/junixsocket-1.3.jar /usr/share/java/$JAVA_LIB
    sudo cp lib-native/libjunixsocket-linux-1.5-amd64.so /usr/lib/

    [ -f $JAVA_LIBS_BASE/$2 ]

    display_ok
}

function install_libobjc2() {
    echo -n "Installing libobjc2..."
    [ -f /usr/local/lib/libobjc.so ] && return
    cd /tmp/
    svn --quiet co http://svn.gna.org/svn/gnustep/libs/libobjc2/trunk libobjc2
    cd libobjc2
    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++  > /dev/null 2>&1
    make -j2 > /dev/null 2>&1
    sudo make install > /dev/null
    display_ok
}

install_libdispatch() {
    echo -n "Installing libdispatch..."
    [ -f /usr/local/lib/libdispatch.so ] && return
    cd /tmp/
    git clone git://github.com/nickhutchinson/libdispatch.git > /dev/null 2>&1
    cd libdispatch/
    sh autogen.sh > /dev/null 2>&1
    ./configure CFLAGS="-I/usr/include/kqueue" LDFLAGS="-lkqueue -lpthread_workqueue -pthread -lm" CC=clang CXX=clang++ > /dev/null 2>&1
    make -j2 > /dev/null 2>&1
    sudo make install > /dev/null
    sudo ldconfig
    display_ok
}

install_gnustep_make() {
    echo -n "Installing gnustep make..."

    fastwget_link "ftp://ftp.gnustep.org/pub/gnustep/core/gnustep-make-2.6.6.tar.gz" "/tmp" "gnustep-make.tar.gz"

    cd /tmp/
    tar -zxvf gnustep-make.tar.gz 1>&3 2>&3
    cd gnustep-make*
    ./configure --with-layout=gnustep CC=clang CXX=clang++ > /dev/null 2>&1
    make -j2 > /dev/null 2>&1
    sudo make install > /dev/null

    . /usr/GNUstep/System/Library/Makefiles/GNUstep.sh

    display_ok
}

install_gnustep_base() {
    echo -n "Installing gnustep base..."

    fastwget_link "ftp://ftp.gnustep.org/pub/gnustep/core/gnustep-base-1.24.6.tar.gz" "/tmp" "gnustep-base.tar.gz"

    cd /tmp/
    tar -zxvf gnustep-base.tar.gz 1>&3 2>&3
    cd gnustep-base*
    export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/include:$LD_LIBRARY_PATH
    ./configure CC=clang CXX=clang++ > /dev/null 2>&1
    make -j2 > /dev/null 2>&1
    sudo bash -c ". /usr/GNUstep/System/Library/Makefiles/GNUstep.sh && make install > /dev/null"

    display_ok
}

install_maven_packages() {
    fastwget_link "https://s3.amazonaws.com/codechecker-install-essentials/FileSearch.zip" "/tmp" "FileSearch.zip"

    cd /tmp
    unzip FileSearch.zip

    cd /tmp/FileSearch
    sudo mvn package
    sudo mvn test
    sudo mvn exec:java -Dexec.mainClass=com.sapient.exercise.FileFinderImpl
}

install_rust_lang() {
    echo -n "Installing rust lang..."
    [ -f /usr/local/bin/rustc ] && return
    curl -s https://static.rust-lang.org/rustup.sh | sudo sh > /dev/null 2>&1
    if [[ $? -ne 0 ]]
    then
        display_error "Error in installing rust lang"
        exit 1
    fi
    display_ok
}

install_android_platform() {
    echo -n "Installing Android platform..."
    if [[ ! -d /mnt/checker/android ]]
    then
        # Setup SDK
        sudo mkdir /mnt/checker/android
        sudo chown -R ubuntu /mnt/checker/android
        cd /mnt/checker/android
        wget_link "http://dl.google.com/android/android-sdk_r23.0.2-linux.tgz" "android-sdk_r23.0.2-linux.tgz"
        tar -zxvf android-sdk_r23.0.2-linux.tgz
        cd android-sdk-linux*
        #sudo expect ~/files/android_install_script.exp
        # Command used to create expect script
        #tools/android update sdk -a --no-ui --filter platform-tools,build-tools-20.0.0,android-20,android-21,extra-android-support,extra-google-webdriver,system-image
        #chmod +r+x -R /mnt/checker/android
    fi
}

function install_osxfuse() {
    echo -n "Installing OSXFUSE..."
    [ -d /Library/Filesystems/osxfusefs.fs ] && return

    wget_link "http://sourceforge.net/projects/osxfuse/files/osxfuse-2.7.0/osxfuse-2.7.0.dmg/download" "/tmp/osxfuse.dmg"
    sudo hdiutil mount /tmp/osxfuse.dmg
    if [[ $? -ne 0 ]]
    then
        display_error "Unable to mount OSXFUSE install package"
        exit 1
    fi

    sudo installer -package "/Volumes/FUSE for OS X/Install OSXFUSE 2.7.pkg" -target "/Volumes/Macintosh HD"
    if [[ $? -ne 0 ]]
    then
        display_error "Unable to install OSXFUSE"
        umount "/Volumes/FUSE for OS X"
        exit 1
    fi

    umount "/Volumes/FUSE for OS X"
    display_ok
}

function install_curl_from_source() {
    echo -n "Installing CURL from source..."
    wget_link "http://curl.haxx.se/download/curl-7.39.0.tar.gz" "/tmp/curl.tar.gz"
    cd /tmp
    tar -zxvf curl.tar.gz 1>&3 2>&3
    cd curl-*

    ./configure > /dev/null 2>&1
    make -j2 > /dev/null 2>&1
    sudo make install > /dev/null
    if [[ $? -ne 0 ]]
    then
        display_error "Unable to install CURL from source"
        exit 1
    fi

    display_ok
}

function install_golang() {
    echo -n "Installing golang..."
    fastwget_link "https://storage.googleapis.com/golang/go1.4.linux-amd64.tar.gz" "/tmp" "golang.tar.gz"
    sudo tar -C /usr/local -xzf /tmp/golang.tar.gz 1>&3 2>&3

    display_ok
}

function make_install_procedure() {
    SOURCE_URL=$1
    SUBSTANCE=$2

    echo -n "Installing $SUBSTANCE from source..."
    pkg-config --libs $SUBSTANCE 1>&3 2>&3
    if [[ $? -eq 0 ]]
    then
        display_ok
        return
    fi

    sudo rm -f /tmp/source.tar.gz
    fastwget_link $1 "/tmp" "source.tar.gz"
    cd /tmp
    tar -zxf source.tar.gz 1>&3 2>&3
    cd $2*

    ./configure > /dev/null 2>&1
    make -j2 > /dev/null 2>&1
    sudo make install > /dev/null

    if [[ $? -ne 0 ]]
    then
        display_error "Unable to install $SUBSTANCE from source"
        exit 1
    fi

    display_ok
}

function install_libconfig_from_source() {
    make_install_procedure "http://www.hyperrealm.com/libconfig/libconfig-1.4.9.tar.gz" "libconfig"
}

function install_libmemcached_from_source() {
    make_install_procedure "https://launchpad.net/libmemcached/1.0/1.0.18/+download/libmemcached-1.0.18.tar.gz" "libmemcached"
}

function install_rlang_packages() {
    aptfast_install r-cran-foreach
    aptfast_install r-cran-base64enc
    aptfast_install r-cran-foreign
    aptfast_install r-cran-bayesm
    aptfast_install r-cran-formula
    aptfast_install r-cran-class
    aptfast_install r-cran-g.data
    aptfast_install r-cran-cluster
    aptfast_install r-cran-numderiv
    aptfast_install r-cran-scales
    aptfast_install r-cran-codetools
    aptfast_install r-cran-permute
    aptfast_install r-cran-date
    aptfast_install r-cran-spatial
    aptfast_install r-cran-psy
    aptfast_install r-cran-digest
    aptfast_install r-cran-hdf5
    aptfast_install r-cran-pwt
    aptfast_install r-cran-statmod
    aptfast_install r-cran-stringr
    aptfast_install r-cran-int64
    aptfast_install r-cran-iterators
    aptfast_install r-cran-lattice
    aptfast_install r-cran-latticeextra
    aptfast_install r-cran-timedate
    aptfast_install r-cran-evaluate
    aptfast_install r-cran-tseries
    aptfast_install r-cran-fastcluster
    aptfast_install r-cran-fbasics
    aptfast_install r-cran-xml
    aptfast_install r-cran-matrix
    aptfast_install r-cran-rjson
    aptfast_install r-cran-zoo
    aptfast_install r-cran-car
    aptfast_install r-cran-plyr
    aptfast_install r-cran-reshape
    aptfast_install r-cran-reshape2
}

#OpenCV Packages
function install_opencv() {
apt-get install build_essential
aptfast_install cmake
aptfast_install git
aptfast_install libgtk2.0-dev
aptfast_install pkg-config
aptfast_install libavcodec-dev
aptfast_install libavformat-dev
aptfast_install libswscale-dev
aptfast_install python-dev 
aptfast_install python-numpy 
aptfast_install libtbb2 
aptfast_install libtbb-dev 
aptfast_install libjpeg-dev 
aptfast_install libpng-dev 
aptfast_install libtiff-dev 
aptfast_install libjasper-dev 
aptfast_install libdc1394-22-dev
aptfast_install ruby

git clone https://github.com/Itseez/opencv.git

cd opencv
mkdir release
cd release

cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..


make -j6
sudo make install/strip

#For Python Language
 apt-get install python-numpy
 apt-get install python-opencv
  apt-get install libcanberra-gtk-module
apt-get install packagekit-gtk3-module

#Ruby
gem install ropencv
}

#Script starts here
if [ "$UNAME" == "Linux" ]; then
    aptget_install aria2
    add_apt_repo "ppa:avsm/ppa"
    add_apt_repo "ppa:hvr/ghc"      #For haskell
    add_apt_repo "ppa:plt/racket"   #For latest version of racket
    add_apt_repo "ppa:staticfloat/juliareleases"    #For latest version of julia
    setup_aptfast
else
    install_osx_essentials
fi
# Upgrade before install
aptget_upgrade

#Install codechecker specific packages
aptfast_install git
if [ "$UNAME" == "Linux" ]; then
    aptfast_install unzip
    aptfast_install g++
    aptfast_install uuid-dev
    aptfast_install make
    aptfast_install libjansson-dev
    aptfast_install libtinyxml-dev
    aptfast_install libcurl4-openssl-dev
    install_libconfig_from_source
    install_libmemcached_from_source
    aptfast_install libmemcached-dev
    aptfast_install liblog4cxx10-dev
    aptfast_install libpoco-dev
    aptfast_install libhiredis-dev
    aptfast_install libjsoncpp-dev
    aptfast_install bindfs
    install_boost_from_source
else
    aptfast_install log4cxx
    aptfast_install tinyxml
    aptfast_install jansson
    aptfast_install libconfig
    aptfast_install libmemcached
    aptfast_install jsoncpp
    aptfast_install hiredis
    aptfast_install pkgconfig
    aptfast_install poco
    aptfast_install wget
    install_boost_from_source
    install_curl_from_source
    install_osxfuse
    aptfast_install bindfs
    aptfast_install rsyslog
    exit 0
fi

install_opencv

exit 0

