------------------------------------------------------------------------------------------------
Automatic Labelling Environment (ALE)
------------------------------------------------------------------------------------------------

Author: Lubor Ladicky

Copyright :
      Lubor Ladicky: lubor@robots.ox.ac.uk
      Philip H.S. Torr:  philiptorr@brookes.ac.uk

Last update :
      17/08/2011

------------------------------------------------------------------------------------------------
Licence
------------------------------------------------------------------------------------------------

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

This software is an environment for pixel-wise labelling problems, designed mainly for object-
class segmentation problem and described in detail in

Lubor Ladicky
Global Structured Models towards Scene Understanding
PhD thesis, Oxford Brookes University, 2011.
 
This software is free ONLY for research purposes. If you want to use any part of the code you
should cite this thesis in any resulting publication. The code remains property of Oxford
Brookes University.

------------------------------------------------------------------------------------------------
External dependencies
------------------------------------------------------------------------------------------------

This software uses max-flow code described in

Yuri Boykov and Vladimir Kolmogorov
An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision.
Transactions on Pattern Analysis and Machine Intelligence, 2004.

This max-flow library is free ONLY for research purposes. If you use this software for research
purposes. You should cite this paper in any resulting publication.

Email : v.kolmogorov@cs.ucl.ac.uk

------------------------------------------------------------------------------------------------

This software uses Developer's Image library (DevIL).

Website : http://openil.sourceforge.net/

The image library is free for any purposes.

------------------------------------------------------------------------------------------------
Related papers
------------------------------------------------------------------------------------------------

This software contains the implementation of the papers below. It also contains
implementation of several other papers and algorithms used and cited in these papers. If you are
using any code corresponding to any of these papers, you should cite them too.

Lubor Ladicky, Chris Russell, Pushmeet Kohli, Philip H.S. Torr
Graph Cut based Inference with Co-occurrence Statistics
Proceedings of the Eleventh European Conference on Computer Vision, 2010. 

Lubor Ladicky, Paul Sturgess, Karteek Alahari, Chris Russell, Philip H.S. Torr
What,Where & How Many? Combining Object Detectors and CRFs
Proceedings of the Eleventh European Conference on Computer Vision, 2010. 

Lubor Ladicky, Paul Sturgess, Chris Russell, Sunando Sengupta, Yalin Bastanlar, William Clocksin,
Philip H.S. Torr
Joint Optimisation for Object Class Segmentation and Dense Stereo Reconstruction
Proceedings British Machine Vision Conference, 2010.

Lubor Ladicky, Chris Russell, Pushmeet Kohli, Philip H.S. Torr
Associative Hierarchical CRFs for Object Class Image Segmentation
Proceedings IEEE Twelfth International Conference on Computer Vision, 2009. 

Chris Russell, Lubor Ladicky, Pushmeet Kohli, Philip H.S. Torr
Exact and Approximate Inference in Associative Hierarchical Networks using Graph Cuts
The 26th Conference on Uncertainty in Artificial Intelligence, 2010.

Pushmeet Kohli, Lubor Ladicky, Philip H.S. Torr
Robust Higher Order Potentials for Enforcing Label Consistency
Proceedings IEEE Conference of Computer Vision and Pattern Recognition, 2008.

------------------------------------------------------------------------------------------------
Usage
------------------------------------------------------------------------------------------------

Copy all the image and ground truth files into the folders set up in dataset.cpp of chosen
data set and run it. By default no command line parameters are necessary.

Current version contains slightly better features than the ones published, thus should lead to
a slightly better performance than reported. Examples of use of other methods in the library are
shown (but not tuned) in the corel data set class. For an import of features, segmentations,
potentials or object detections from other executable, use the Dummy classes provided.

------------------------------------------------------------------------------------------------
Installation
------------------------------------------------------------------------------------------------

Current version contains a project file for Visual Studio in MS Windows and a makefile for
linux compiler g++. Windows version is self-contained, linux requires DevIL library to be
installed. You can install it in Ubuntu od Debian distributions by typing : 
"sudo apt-get install libdevil-dev" or in Suse or ReHat distributions using command
"yum install libdevil-dev". For other linux version you probably know what to do:). The code
has been tested in Kubuntu and RedHat. If there are any problems in other distributions, please
report the problem.

------------------------------------------------------------------------------------------------
Bug reports
------------------------------------------------------------------------------------------------

Please report all bugs to the e-mail address above.
