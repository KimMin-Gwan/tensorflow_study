ыю
мї
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Ѕ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И®
@
StaticRegexFullMatch	
input

output
"
patternstring
ч
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.11.02v2.11.0-rc2-15-g6290819256d8јИ
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
Ф
Adam/v/sequential/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/v/sequential/dense_7/bias
Н
2Adam/v/sequential/dense_7/bias/Read/ReadVariableOpReadVariableOpAdam/v/sequential/dense_7/bias*
_output_shapes
:*
dtype0
Ф
Adam/m/sequential/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/m/sequential/dense_7/bias
Н
2Adam/m/sequential/dense_7/bias/Read/ReadVariableOpReadVariableOpAdam/m/sequential/dense_7/bias*
_output_shapes
:*
dtype0
Э
 Adam/v/sequential/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*1
shared_name" Adam/v/sequential/dense_7/kernel
Ц
4Adam/v/sequential/dense_7/kernel/Read/ReadVariableOpReadVariableOp Adam/v/sequential/dense_7/kernel*
_output_shapes
:	А*
dtype0
Э
 Adam/m/sequential/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*1
shared_name" Adam/m/sequential/dense_7/kernel
Ц
4Adam/m/sequential/dense_7/kernel/Read/ReadVariableOpReadVariableOp Adam/m/sequential/dense_7/kernel*
_output_shapes
:	А*
dtype0
Х
Adam/v/sequential/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*/
shared_name Adam/v/sequential/dense_6/bias
О
2Adam/v/sequential/dense_6/bias/Read/ReadVariableOpReadVariableOpAdam/v/sequential/dense_6/bias*
_output_shapes	
:А*
dtype0
Х
Adam/m/sequential/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*/
shared_name Adam/m/sequential/dense_6/bias
О
2Adam/m/sequential/dense_6/bias/Read/ReadVariableOpReadVariableOpAdam/m/sequential/dense_6/bias*
_output_shapes	
:А*
dtype0
Ю
 Adam/v/sequential/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*1
shared_name" Adam/v/sequential/dense_6/kernel
Ч
4Adam/v/sequential/dense_6/kernel/Read/ReadVariableOpReadVariableOp Adam/v/sequential/dense_6/kernel* 
_output_shapes
:
АА*
dtype0
Ю
 Adam/m/sequential/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*1
shared_name" Adam/m/sequential/dense_6/kernel
Ч
4Adam/m/sequential/dense_6/kernel/Read/ReadVariableOpReadVariableOp Adam/m/sequential/dense_6/kernel* 
_output_shapes
:
АА*
dtype0
Х
Adam/v/sequential/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*/
shared_name Adam/v/sequential/dense_5/bias
О
2Adam/v/sequential/dense_5/bias/Read/ReadVariableOpReadVariableOpAdam/v/sequential/dense_5/bias*
_output_shapes	
:А*
dtype0
Х
Adam/m/sequential/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*/
shared_name Adam/m/sequential/dense_5/bias
О
2Adam/m/sequential/dense_5/bias/Read/ReadVariableOpReadVariableOpAdam/m/sequential/dense_5/bias*
_output_shapes	
:А*
dtype0
Ю
 Adam/v/sequential/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*1
shared_name" Adam/v/sequential/dense_5/kernel
Ч
4Adam/v/sequential/dense_5/kernel/Read/ReadVariableOpReadVariableOp Adam/v/sequential/dense_5/kernel* 
_output_shapes
:
АА*
dtype0
Ю
 Adam/m/sequential/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*1
shared_name" Adam/m/sequential/dense_5/kernel
Ч
4Adam/m/sequential/dense_5/kernel/Read/ReadVariableOpReadVariableOp Adam/m/sequential/dense_5/kernel* 
_output_shapes
:
АА*
dtype0
Х
Adam/v/sequential/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*/
shared_name Adam/v/sequential/dense_4/bias
О
2Adam/v/sequential/dense_4/bias/Read/ReadVariableOpReadVariableOpAdam/v/sequential/dense_4/bias*
_output_shapes	
:А*
dtype0
Х
Adam/m/sequential/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*/
shared_name Adam/m/sequential/dense_4/bias
О
2Adam/m/sequential/dense_4/bias/Read/ReadVariableOpReadVariableOpAdam/m/sequential/dense_4/bias*
_output_shapes	
:А*
dtype0
Ю
 Adam/v/sequential/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*1
shared_name" Adam/v/sequential/dense_4/kernel
Ч
4Adam/v/sequential/dense_4/kernel/Read/ReadVariableOpReadVariableOp Adam/v/sequential/dense_4/kernel* 
_output_shapes
:
АА*
dtype0
Ю
 Adam/m/sequential/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*1
shared_name" Adam/m/sequential/dense_4/kernel
Ч
4Adam/m/sequential/dense_4/kernel/Read/ReadVariableOpReadVariableOp Adam/m/sequential/dense_4/kernel* 
_output_shapes
:
АА*
dtype0
Х
Adam/v/sequential/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*/
shared_name Adam/v/sequential/dense_3/bias
О
2Adam/v/sequential/dense_3/bias/Read/ReadVariableOpReadVariableOpAdam/v/sequential/dense_3/bias*
_output_shapes	
:А*
dtype0
Х
Adam/m/sequential/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*/
shared_name Adam/m/sequential/dense_3/bias
О
2Adam/m/sequential/dense_3/bias/Read/ReadVariableOpReadVariableOpAdam/m/sequential/dense_3/bias*
_output_shapes	
:А*
dtype0
Э
 Adam/v/sequential/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*1
shared_name" Adam/v/sequential/dense_3/kernel
Ц
4Adam/v/sequential/dense_3/kernel/Read/ReadVariableOpReadVariableOp Adam/v/sequential/dense_3/kernel*
_output_shapes
:	@А*
dtype0
Э
 Adam/m/sequential/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А*1
shared_name" Adam/m/sequential/dense_3/kernel
Ц
4Adam/m/sequential/dense_3/kernel/Read/ReadVariableOpReadVariableOp Adam/m/sequential/dense_3/kernel*
_output_shapes
:	@А*
dtype0
Ф
Adam/v/sequential/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/v/sequential/dense_2/bias
Н
2Adam/v/sequential/dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/sequential/dense_2/bias*
_output_shapes
:@*
dtype0
Ф
Adam/m/sequential/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/m/sequential/dense_2/bias
Н
2Adam/m/sequential/dense_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/sequential/dense_2/bias*
_output_shapes
:@*
dtype0
Ь
 Adam/v/sequential/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*1
shared_name" Adam/v/sequential/dense_2/kernel
Х
4Adam/v/sequential/dense_2/kernel/Read/ReadVariableOpReadVariableOp Adam/v/sequential/dense_2/kernel*
_output_shapes

:@@*
dtype0
Ь
 Adam/m/sequential/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*1
shared_name" Adam/m/sequential/dense_2/kernel
Х
4Adam/m/sequential/dense_2/kernel/Read/ReadVariableOpReadVariableOp Adam/m/sequential/dense_2/kernel*
_output_shapes

:@@*
dtype0
Ф
Adam/v/sequential/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/v/sequential/dense_1/bias
Н
2Adam/v/sequential/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/sequential/dense_1/bias*
_output_shapes
:@*
dtype0
Ф
Adam/m/sequential/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/m/sequential/dense_1/bias
Н
2Adam/m/sequential/dense_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/sequential/dense_1/bias*
_output_shapes
:@*
dtype0
Ь
 Adam/v/sequential/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*1
shared_name" Adam/v/sequential/dense_1/kernel
Х
4Adam/v/sequential/dense_1/kernel/Read/ReadVariableOpReadVariableOp Adam/v/sequential/dense_1/kernel*
_output_shapes

:@@*
dtype0
Ь
 Adam/m/sequential/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*1
shared_name" Adam/m/sequential/dense_1/kernel
Х
4Adam/m/sequential/dense_1/kernel/Read/ReadVariableOpReadVariableOp Adam/m/sequential/dense_1/kernel*
_output_shapes

:@@*
dtype0
Р
Adam/v/sequential/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_nameAdam/v/sequential/dense/bias
Й
0Adam/v/sequential/dense/bias/Read/ReadVariableOpReadVariableOpAdam/v/sequential/dense/bias*
_output_shapes
:@*
dtype0
Р
Adam/m/sequential/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_nameAdam/m/sequential/dense/bias
Й
0Adam/m/sequential/dense/bias/Read/ReadVariableOpReadVariableOpAdam/m/sequential/dense/bias*
_output_shapes
:@*
dtype0
Ш
Adam/v/sequential/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*/
shared_name Adam/v/sequential/dense/kernel
С
2Adam/v/sequential/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/v/sequential/dense/kernel*
_output_shapes

:@*
dtype0
Ш
Adam/m/sequential/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*/
shared_name Adam/m/sequential/dense/kernel
С
2Adam/m/sequential/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/m/sequential/dense/kernel*
_output_shapes

:@*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
Ж
sequential/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namesequential/dense_7/bias

+sequential/dense_7/bias/Read/ReadVariableOpReadVariableOpsequential/dense_7/bias*
_output_shapes
:*
dtype0
П
sequential/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А**
shared_namesequential/dense_7/kernel
И
-sequential/dense_7/kernel/Read/ReadVariableOpReadVariableOpsequential/dense_7/kernel*
_output_shapes
:	А*
dtype0
З
sequential/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_namesequential/dense_6/bias
А
+sequential/dense_6/bias/Read/ReadVariableOpReadVariableOpsequential/dense_6/bias*
_output_shapes	
:А*
dtype0
Р
sequential/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА**
shared_namesequential/dense_6/kernel
Й
-sequential/dense_6/kernel/Read/ReadVariableOpReadVariableOpsequential/dense_6/kernel* 
_output_shapes
:
АА*
dtype0
З
sequential/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_namesequential/dense_5/bias
А
+sequential/dense_5/bias/Read/ReadVariableOpReadVariableOpsequential/dense_5/bias*
_output_shapes	
:А*
dtype0
Р
sequential/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА**
shared_namesequential/dense_5/kernel
Й
-sequential/dense_5/kernel/Read/ReadVariableOpReadVariableOpsequential/dense_5/kernel* 
_output_shapes
:
АА*
dtype0
З
sequential/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_namesequential/dense_4/bias
А
+sequential/dense_4/bias/Read/ReadVariableOpReadVariableOpsequential/dense_4/bias*
_output_shapes	
:А*
dtype0
Р
sequential/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА**
shared_namesequential/dense_4/kernel
Й
-sequential/dense_4/kernel/Read/ReadVariableOpReadVariableOpsequential/dense_4/kernel* 
_output_shapes
:
АА*
dtype0
З
sequential/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*(
shared_namesequential/dense_3/bias
А
+sequential/dense_3/bias/Read/ReadVariableOpReadVariableOpsequential/dense_3/bias*
_output_shapes	
:А*
dtype0
П
sequential/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@А**
shared_namesequential/dense_3/kernel
И
-sequential/dense_3/kernel/Read/ReadVariableOpReadVariableOpsequential/dense_3/kernel*
_output_shapes
:	@А*
dtype0
Ж
sequential/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_namesequential/dense_2/bias

+sequential/dense_2/bias/Read/ReadVariableOpReadVariableOpsequential/dense_2/bias*
_output_shapes
:@*
dtype0
О
sequential/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@**
shared_namesequential/dense_2/kernel
З
-sequential/dense_2/kernel/Read/ReadVariableOpReadVariableOpsequential/dense_2/kernel*
_output_shapes

:@@*
dtype0
Ж
sequential/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_namesequential/dense_1/bias

+sequential/dense_1/bias/Read/ReadVariableOpReadVariableOpsequential/dense_1/bias*
_output_shapes
:@*
dtype0
О
sequential/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@**
shared_namesequential/dense_1/kernel
З
-sequential/dense_1/kernel/Read/ReadVariableOpReadVariableOpsequential/dense_1/kernel*
_output_shapes

:@@*
dtype0
В
sequential/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_namesequential/dense/bias
{
)sequential/dense/bias/Read/ReadVariableOpReadVariableOpsequential/dense/bias*
_output_shapes
:@*
dtype0
К
sequential/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*(
shared_namesequential/dense/kernel
Г
+sequential/dense/kernel/Read/ReadVariableOpReadVariableOpsequential/dense/kernel*
_output_shapes

:@*
dtype0
m
serving_default_IDPlaceholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
w
serving_default_acousticnessPlaceholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
w
serving_default_danceabilityPlaceholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
s
serving_default_durationPlaceholder*#
_output_shapes
:€€€€€€€€€*
dtype0	*
shape:€€€€€€€€€
q
serving_default_energyPlaceholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
{
 serving_default_instrumentalnessPlaceholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
n
serving_default_keyPlaceholder*#
_output_shapes
:€€€€€€€€€*
dtype0	*
shape:€€€€€€€€€
s
serving_default_livenessPlaceholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
s
serving_default_loudnessPlaceholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
v
serving_default_speechinessPlaceholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
p
serving_default_tempoPlaceholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
r
serving_default_valencePlaceholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
Ъ
StatefulPartitionedCallStatefulPartitionedCallserving_default_IDserving_default_acousticnessserving_default_danceabilityserving_default_durationserving_default_energy serving_default_instrumentalnessserving_default_keyserving_default_livenessserving_default_loudnessserving_default_speechinessserving_default_temposerving_default_valencesequential/dense/kernelsequential/dense/biassequential/dense_1/kernelsequential/dense_1/biassequential/dense_2/kernelsequential/dense_2/biassequential/dense_3/kernelsequential/dense_3/biassequential/dense_4/kernelsequential/dense_4/biassequential/dense_5/kernelsequential/dense_5/biassequential/dense_6/kernelsequential/dense_6/biassequential/dense_7/kernelsequential/dense_7/bias*'
Tin 
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*2
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *-
f(R&
$__inference_signature_wrapper_989035

NoOpNoOp
≤k
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*нj
valueгjBаj Bўj
ґ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
_build_input_shape

signatures*
і
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_feature_columns

_resources* 
¶
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias*
¶
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

+kernel
,bias*
¶
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias*
¶
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

;kernel
<bias*
¶
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses

Ckernel
Dbias*
¶
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses

Kkernel
Lbias*
•
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses
S_random_generator* 
¶
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses

Zkernel
[bias*
¶
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses

bkernel
cbias*
z
#0
$1
+2
,3
34
45
;6
<7
C8
D9
K10
L11
Z12
[13
b14
c15*
z
#0
$1
+2
,3
34
45
;6
<7
C8
D9
K10
L11
Z12
[13
b14
c15*
* 
∞
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
itrace_0
jtrace_1
ktrace_2
ltrace_3* 
6
mtrace_0
ntrace_1
otrace_2
ptrace_3* 
* 
Б
q
_variables
r_iterations
s_learning_rate
t_index_dict
u
_momentums
v_velocities
w_update_step_xla*
* 

xserving_default* 
* 
* 
* 
С
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

~trace_0
trace_1* 

Аtrace_0
Бtrace_1* 
* 
* 

#0
$1*

#0
$1*
* 
Ш
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

Зtrace_0* 

Иtrace_0* 
ga
VARIABLE_VALUEsequential/dense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEsequential/dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

+0
,1*

+0
,1*
* 
Ш
Йnon_trainable_variables
Кlayers
Лmetrics
 Мlayer_regularization_losses
Нlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses*

Оtrace_0* 

Пtrace_0* 
ic
VARIABLE_VALUEsequential/dense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEsequential/dense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

30
41*

30
41*
* 
Ш
Рnon_trainable_variables
Сlayers
Тmetrics
 Уlayer_regularization_losses
Фlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*

Хtrace_0* 

Цtrace_0* 
ic
VARIABLE_VALUEsequential/dense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEsequential/dense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

;0
<1*

;0
<1*
* 
Ш
Чnon_trainable_variables
Шlayers
Щmetrics
 Ъlayer_regularization_losses
Ыlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*

Ьtrace_0* 

Эtrace_0* 
ic
VARIABLE_VALUEsequential/dense_3/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEsequential/dense_3/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

C0
D1*

C0
D1*
* 
Ш
Юnon_trainable_variables
Яlayers
†metrics
 °layer_regularization_losses
Ґlayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses*

£trace_0* 

§trace_0* 
ic
VARIABLE_VALUEsequential/dense_4/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEsequential/dense_4/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

K0
L1*

K0
L1*
* 
Ш
•non_trainable_variables
¶layers
Іmetrics
 ®layer_regularization_losses
©layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses*

™trace_0* 

Ђtrace_0* 
ic
VARIABLE_VALUEsequential/dense_5/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEsequential/dense_5/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ц
ђnon_trainable_variables
≠layers
Ѓmetrics
 ѓlayer_regularization_losses
∞layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses* 

±trace_0
≤trace_1* 

≥trace_0
іtrace_1* 
* 

Z0
[1*

Z0
[1*
* 
Ш
µnon_trainable_variables
ґlayers
Јmetrics
 Єlayer_regularization_losses
єlayer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses*

Їtrace_0* 

їtrace_0* 
ic
VARIABLE_VALUEsequential/dense_6/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEsequential/dense_6/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

b0
c1*

b0
c1*
* 
Ш
Љnon_trainable_variables
љlayers
Њmetrics
 њlayer_regularization_losses
јlayer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses*

Ѕtrace_0* 

¬trace_0* 
ic
VARIABLE_VALUEsequential/dense_7/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEsequential/dense_7/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
J
0
1
2
3
4
5
6
7
	8

9*

√0
ƒ1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
Ґ
r0
≈1
∆2
«3
»4
…5
 6
Ћ7
ћ8
Ќ9
ќ10
ѕ11
–12
—13
“14
”15
‘16
’17
÷18
„19
Ў20
ў21
Џ22
џ23
№24
Ё25
ё26
я27
а28
б29
в30
г31
д32*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
К
≈0
«1
…2
Ћ3
Ќ4
ѕ5
—6
”7
’8
„9
ў10
џ11
Ё12
я13
б14
г15*
К
∆0
»1
 2
ћ3
ќ4
–5
“6
‘7
÷8
Ў9
Џ10
№11
ё12
а13
в14
д15*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
е	variables
ж	keras_api

зtotal

иcount*
M
й	variables
к	keras_api

лtotal

мcount
н
_fn_kwargs*
ic
VARIABLE_VALUEAdam/m/sequential/dense/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/v/sequential/dense/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/m/sequential/dense/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEAdam/v/sequential/dense/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/m/sequential/dense_1/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/v/sequential/dense_1/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/m/sequential/dense_1/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/v/sequential/dense_1/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE Adam/m/sequential/dense_2/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/v/sequential/dense_2/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/sequential/dense_2/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/sequential/dense_2/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/m/sequential/dense_3/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/v/sequential/dense_3/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/sequential/dense_3/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/sequential/dense_3/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/m/sequential/dense_4/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/v/sequential/dense_4/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/sequential/dense_4/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/sequential/dense_4/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/m/sequential/dense_5/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/v/sequential/dense_5/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/sequential/dense_5/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/sequential/dense_5/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/m/sequential/dense_6/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/v/sequential/dense_6/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/sequential/dense_6/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/sequential/dense_6/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/m/sequential/dense_7/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE Adam/v/sequential/dense_7/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/m/sequential/dense_7/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEAdam/v/sequential/dense_7/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*

з0
и1*

е	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

л0
м1*

й	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ч
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+sequential/dense/kernel/Read/ReadVariableOp)sequential/dense/bias/Read/ReadVariableOp-sequential/dense_1/kernel/Read/ReadVariableOp+sequential/dense_1/bias/Read/ReadVariableOp-sequential/dense_2/kernel/Read/ReadVariableOp+sequential/dense_2/bias/Read/ReadVariableOp-sequential/dense_3/kernel/Read/ReadVariableOp+sequential/dense_3/bias/Read/ReadVariableOp-sequential/dense_4/kernel/Read/ReadVariableOp+sequential/dense_4/bias/Read/ReadVariableOp-sequential/dense_5/kernel/Read/ReadVariableOp+sequential/dense_5/bias/Read/ReadVariableOp-sequential/dense_6/kernel/Read/ReadVariableOp+sequential/dense_6/bias/Read/ReadVariableOp-sequential/dense_7/kernel/Read/ReadVariableOp+sequential/dense_7/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp2Adam/m/sequential/dense/kernel/Read/ReadVariableOp2Adam/v/sequential/dense/kernel/Read/ReadVariableOp0Adam/m/sequential/dense/bias/Read/ReadVariableOp0Adam/v/sequential/dense/bias/Read/ReadVariableOp4Adam/m/sequential/dense_1/kernel/Read/ReadVariableOp4Adam/v/sequential/dense_1/kernel/Read/ReadVariableOp2Adam/m/sequential/dense_1/bias/Read/ReadVariableOp2Adam/v/sequential/dense_1/bias/Read/ReadVariableOp4Adam/m/sequential/dense_2/kernel/Read/ReadVariableOp4Adam/v/sequential/dense_2/kernel/Read/ReadVariableOp2Adam/m/sequential/dense_2/bias/Read/ReadVariableOp2Adam/v/sequential/dense_2/bias/Read/ReadVariableOp4Adam/m/sequential/dense_3/kernel/Read/ReadVariableOp4Adam/v/sequential/dense_3/kernel/Read/ReadVariableOp2Adam/m/sequential/dense_3/bias/Read/ReadVariableOp2Adam/v/sequential/dense_3/bias/Read/ReadVariableOp4Adam/m/sequential/dense_4/kernel/Read/ReadVariableOp4Adam/v/sequential/dense_4/kernel/Read/ReadVariableOp2Adam/m/sequential/dense_4/bias/Read/ReadVariableOp2Adam/v/sequential/dense_4/bias/Read/ReadVariableOp4Adam/m/sequential/dense_5/kernel/Read/ReadVariableOp4Adam/v/sequential/dense_5/kernel/Read/ReadVariableOp2Adam/m/sequential/dense_5/bias/Read/ReadVariableOp2Adam/v/sequential/dense_5/bias/Read/ReadVariableOp4Adam/m/sequential/dense_6/kernel/Read/ReadVariableOp4Adam/v/sequential/dense_6/kernel/Read/ReadVariableOp2Adam/m/sequential/dense_6/bias/Read/ReadVariableOp2Adam/v/sequential/dense_6/bias/Read/ReadVariableOp4Adam/m/sequential/dense_7/kernel/Read/ReadVariableOp4Adam/v/sequential/dense_7/kernel/Read/ReadVariableOp2Adam/m/sequential/dense_7/bias/Read/ReadVariableOp2Adam/v/sequential/dense_7/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*C
Tin<
:28	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *(
f#R!
__inference__traced_save_990201
Ї
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamesequential/dense/kernelsequential/dense/biassequential/dense_1/kernelsequential/dense_1/biassequential/dense_2/kernelsequential/dense_2/biassequential/dense_3/kernelsequential/dense_3/biassequential/dense_4/kernelsequential/dense_4/biassequential/dense_5/kernelsequential/dense_5/biassequential/dense_6/kernelsequential/dense_6/biassequential/dense_7/kernelsequential/dense_7/bias	iterationlearning_rateAdam/m/sequential/dense/kernelAdam/v/sequential/dense/kernelAdam/m/sequential/dense/biasAdam/v/sequential/dense/bias Adam/m/sequential/dense_1/kernel Adam/v/sequential/dense_1/kernelAdam/m/sequential/dense_1/biasAdam/v/sequential/dense_1/bias Adam/m/sequential/dense_2/kernel Adam/v/sequential/dense_2/kernelAdam/m/sequential/dense_2/biasAdam/v/sequential/dense_2/bias Adam/m/sequential/dense_3/kernel Adam/v/sequential/dense_3/kernelAdam/m/sequential/dense_3/biasAdam/v/sequential/dense_3/bias Adam/m/sequential/dense_4/kernel Adam/v/sequential/dense_4/kernelAdam/m/sequential/dense_4/biasAdam/v/sequential/dense_4/bias Adam/m/sequential/dense_5/kernel Adam/v/sequential/dense_5/kernelAdam/m/sequential/dense_5/biasAdam/v/sequential/dense_5/bias Adam/m/sequential/dense_6/kernel Adam/v/sequential/dense_6/kernelAdam/m/sequential/dense_6/biasAdam/v/sequential/dense_6/bias Adam/m/sequential/dense_7/kernel Adam/v/sequential/dense_7/kernelAdam/m/sequential/dense_7/biasAdam/v/sequential/dense_7/biastotal_1count_1totalcount*B
Tin;
927*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference__traced_restore_990373Хм
с
a
(__inference_dropout_layer_call_fn_989948

inputs
identityИҐStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_988416p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ґи
џ$
"__inference__traced_restore_990373
file_prefix:
(assignvariableop_sequential_dense_kernel:@6
(assignvariableop_1_sequential_dense_bias:@>
,assignvariableop_2_sequential_dense_1_kernel:@@8
*assignvariableop_3_sequential_dense_1_bias:@>
,assignvariableop_4_sequential_dense_2_kernel:@@8
*assignvariableop_5_sequential_dense_2_bias:@?
,assignvariableop_6_sequential_dense_3_kernel:	@А9
*assignvariableop_7_sequential_dense_3_bias:	А@
,assignvariableop_8_sequential_dense_4_kernel:
АА9
*assignvariableop_9_sequential_dense_4_bias:	АA
-assignvariableop_10_sequential_dense_5_kernel:
АА:
+assignvariableop_11_sequential_dense_5_bias:	АA
-assignvariableop_12_sequential_dense_6_kernel:
АА:
+assignvariableop_13_sequential_dense_6_bias:	А@
-assignvariableop_14_sequential_dense_7_kernel:	А9
+assignvariableop_15_sequential_dense_7_bias:'
assignvariableop_16_iteration:	 +
!assignvariableop_17_learning_rate: D
2assignvariableop_18_adam_m_sequential_dense_kernel:@D
2assignvariableop_19_adam_v_sequential_dense_kernel:@>
0assignvariableop_20_adam_m_sequential_dense_bias:@>
0assignvariableop_21_adam_v_sequential_dense_bias:@F
4assignvariableop_22_adam_m_sequential_dense_1_kernel:@@F
4assignvariableop_23_adam_v_sequential_dense_1_kernel:@@@
2assignvariableop_24_adam_m_sequential_dense_1_bias:@@
2assignvariableop_25_adam_v_sequential_dense_1_bias:@F
4assignvariableop_26_adam_m_sequential_dense_2_kernel:@@F
4assignvariableop_27_adam_v_sequential_dense_2_kernel:@@@
2assignvariableop_28_adam_m_sequential_dense_2_bias:@@
2assignvariableop_29_adam_v_sequential_dense_2_bias:@G
4assignvariableop_30_adam_m_sequential_dense_3_kernel:	@АG
4assignvariableop_31_adam_v_sequential_dense_3_kernel:	@АA
2assignvariableop_32_adam_m_sequential_dense_3_bias:	АA
2assignvariableop_33_adam_v_sequential_dense_3_bias:	АH
4assignvariableop_34_adam_m_sequential_dense_4_kernel:
ААH
4assignvariableop_35_adam_v_sequential_dense_4_kernel:
ААA
2assignvariableop_36_adam_m_sequential_dense_4_bias:	АA
2assignvariableop_37_adam_v_sequential_dense_4_bias:	АH
4assignvariableop_38_adam_m_sequential_dense_5_kernel:
ААH
4assignvariableop_39_adam_v_sequential_dense_5_kernel:
ААA
2assignvariableop_40_adam_m_sequential_dense_5_bias:	АA
2assignvariableop_41_adam_v_sequential_dense_5_bias:	АH
4assignvariableop_42_adam_m_sequential_dense_6_kernel:
ААH
4assignvariableop_43_adam_v_sequential_dense_6_kernel:
ААA
2assignvariableop_44_adam_m_sequential_dense_6_bias:	АA
2assignvariableop_45_adam_v_sequential_dense_6_bias:	АG
4assignvariableop_46_adam_m_sequential_dense_7_kernel:	АG
4assignvariableop_47_adam_v_sequential_dense_7_kernel:	А@
2assignvariableop_48_adam_m_sequential_dense_7_bias:@
2assignvariableop_49_adam_v_sequential_dense_7_bias:%
assignvariableop_50_total_1: %
assignvariableop_51_count_1: #
assignvariableop_52_total: #
assignvariableop_53_count: 
identity_55ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9≥
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*ў
valueѕBћ7B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHя
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*Б
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B і
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*т
_output_shapesя
№:::::::::::::::::::::::::::::::::::::::::::::::::::::::*E
dtypes;
927	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOpAssignVariableOp(assignvariableop_sequential_dense_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:њ
AssignVariableOp_1AssignVariableOp(assignvariableop_1_sequential_dense_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_2AssignVariableOp,assignvariableop_2_sequential_dense_1_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_3AssignVariableOp*assignvariableop_3_sequential_dense_1_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_4AssignVariableOp,assignvariableop_4_sequential_dense_2_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_5AssignVariableOp*assignvariableop_5_sequential_dense_2_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_6AssignVariableOp,assignvariableop_6_sequential_dense_3_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_7AssignVariableOp*assignvariableop_7_sequential_dense_3_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:√
AssignVariableOp_8AssignVariableOp,assignvariableop_8_sequential_dense_4_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Ѕ
AssignVariableOp_9AssignVariableOp*assignvariableop_9_sequential_dense_4_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:∆
AssignVariableOp_10AssignVariableOp-assignvariableop_10_sequential_dense_5_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_11AssignVariableOp+assignvariableop_11_sequential_dense_5_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:∆
AssignVariableOp_12AssignVariableOp-assignvariableop_12_sequential_dense_6_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_13AssignVariableOp+assignvariableop_13_sequential_dense_6_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:∆
AssignVariableOp_14AssignVariableOp-assignvariableop_14_sequential_dense_7_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:ƒ
AssignVariableOp_15AssignVariableOp+assignvariableop_15_sequential_dense_7_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:ґ
AssignVariableOp_16AssignVariableOpassignvariableop_16_iterationIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_17AssignVariableOp!assignvariableop_17_learning_rateIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_18AssignVariableOp2assignvariableop_18_adam_m_sequential_dense_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_19AssignVariableOp2assignvariableop_19_adam_v_sequential_dense_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:…
AssignVariableOp_20AssignVariableOp0assignvariableop_20_adam_m_sequential_dense_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:…
AssignVariableOp_21AssignVariableOp0assignvariableop_21_adam_v_sequential_dense_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_22AssignVariableOp4assignvariableop_22_adam_m_sequential_dense_1_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_23AssignVariableOp4assignvariableop_23_adam_v_sequential_dense_1_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_24AssignVariableOp2assignvariableop_24_adam_m_sequential_dense_1_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_25AssignVariableOp2assignvariableop_25_adam_v_sequential_dense_1_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_26AssignVariableOp4assignvariableop_26_adam_m_sequential_dense_2_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_27AssignVariableOp4assignvariableop_27_adam_v_sequential_dense_2_kernelIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_28AssignVariableOp2assignvariableop_28_adam_m_sequential_dense_2_biasIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_29AssignVariableOp2assignvariableop_29_adam_v_sequential_dense_2_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_30AssignVariableOp4assignvariableop_30_adam_m_sequential_dense_3_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_31AssignVariableOp4assignvariableop_31_adam_v_sequential_dense_3_kernelIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_32AssignVariableOp2assignvariableop_32_adam_m_sequential_dense_3_biasIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_33AssignVariableOp2assignvariableop_33_adam_v_sequential_dense_3_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_34AssignVariableOp4assignvariableop_34_adam_m_sequential_dense_4_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_35AssignVariableOp4assignvariableop_35_adam_v_sequential_dense_4_kernelIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_36AssignVariableOp2assignvariableop_36_adam_m_sequential_dense_4_biasIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_37AssignVariableOp2assignvariableop_37_adam_v_sequential_dense_4_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_38AssignVariableOp4assignvariableop_38_adam_m_sequential_dense_5_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_39AssignVariableOp4assignvariableop_39_adam_v_sequential_dense_5_kernelIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_40AssignVariableOp2assignvariableop_40_adam_m_sequential_dense_5_biasIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_41AssignVariableOp2assignvariableop_41_adam_v_sequential_dense_5_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_42AssignVariableOp4assignvariableop_42_adam_m_sequential_dense_6_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_43AssignVariableOp4assignvariableop_43_adam_v_sequential_dense_6_kernelIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_44AssignVariableOp2assignvariableop_44_adam_m_sequential_dense_6_biasIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_45AssignVariableOp2assignvariableop_45_adam_v_sequential_dense_6_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_46AssignVariableOp4assignvariableop_46_adam_m_sequential_dense_7_kernelIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_47AssignVariableOp4assignvariableop_47_adam_v_sequential_dense_7_kernelIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_48AssignVariableOp2assignvariableop_48_adam_m_sequential_dense_7_biasIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_49AssignVariableOp2assignvariableop_49_adam_v_sequential_dense_7_biasIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_50AssignVariableOpassignvariableop_50_total_1Identity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_51AssignVariableOpassignvariableop_51_count_1Identity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOp_52AssignVariableOpassignvariableop_52_totalIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:≤
AssignVariableOp_53AssignVariableOpassignvariableop_53_countIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 у	
Identity_54Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_55IdentityIdentity_54:output:0^NoOp_1*
T0*
_output_shapes
: а	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_55Identity_55:output:0*Б
_input_shapesp
n: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¶

ч
C__inference_dense_4_layer_call_and_return_conditional_losses_988276

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Љ
У
&__inference_dense_layer_call_fn_989827

inputs
unknown:@
	unknown_0:@
identityИҐStatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_988208o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Џ
a
C__inference_dropout_layer_call_and_return_conditional_losses_988304

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
чА
Ъ
J__inference_dense_features_layer_call_and_return_conditional_losses_988638
features

features_1

features_2

features_3	

features_4

features_5

features_6	

features_7

features_8

features_9
features_10
features_11
identityf
acousticness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Й
acousticness/ExpandDims
ExpandDims
features_1$acousticness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€b
acousticness/ShapeShape acousticness/ExpandDims:output:0*
T0*
_output_shapes
:j
 acousticness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"acousticness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"acousticness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
acousticness/strided_sliceStridedSliceacousticness/Shape:output:0)acousticness/strided_slice/stack:output:0+acousticness/strided_slice/stack_1:output:0+acousticness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
acousticness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ь
acousticness/Reshape/shapePack#acousticness/strided_slice:output:0%acousticness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ш
acousticness/ReshapeReshape acousticness/ExpandDims:output:0#acousticness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€f
danceability/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Й
danceability/ExpandDims
ExpandDims
features_2$danceability/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€b
danceability/ShapeShape danceability/ExpandDims:output:0*
T0*
_output_shapes
:j
 danceability/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"danceability/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"danceability/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
danceability/strided_sliceStridedSlicedanceability/Shape:output:0)danceability/strided_slice/stack:output:0+danceability/strided_slice/stack_1:output:0+danceability/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
danceability/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ь
danceability/Reshape/shapePack#danceability/strided_slice:output:0%danceability/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ш
danceability/ReshapeReshape danceability/ExpandDims:output:0#danceability/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€b
duration/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Б
duration/ExpandDims
ExpandDims
features_3 duration/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:€€€€€€€€€t
duration/CastCastduration/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:€€€€€€€€€O
duration/ShapeShapeduration/Cast:y:0*
T0*
_output_shapes
:f
duration/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
duration/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
duration/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
duration/strided_sliceStridedSliceduration/Shape:output:0%duration/strided_slice/stack:output:0'duration/strided_slice/stack_1:output:0'duration/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
duration/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Р
duration/Reshape/shapePackduration/strided_slice:output:0!duration/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Б
duration/ReshapeReshapeduration/Cast:y:0duration/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
energy/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€}
energy/ExpandDims
ExpandDims
features_4energy/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
energy/ShapeShapeenergy/ExpandDims:output:0*
T0*
_output_shapes
:d
energy/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
energy/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
energy/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ф
energy/strided_sliceStridedSliceenergy/Shape:output:0#energy/strided_slice/stack:output:0%energy/strided_slice/stack_1:output:0%energy/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
energy/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :К
energy/Reshape/shapePackenergy/strided_slice:output:0energy/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ж
energy/ReshapeReshapeenergy/ExpandDims:output:0energy/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€j
instrumentalness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€С
instrumentalness/ExpandDims
ExpandDims
features_5(instrumentalness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€j
instrumentalness/ShapeShape$instrumentalness/ExpandDims:output:0*
T0*
_output_shapes
:n
$instrumentalness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&instrumentalness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&instrumentalness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¶
instrumentalness/strided_sliceStridedSliceinstrumentalness/Shape:output:0-instrumentalness/strided_slice/stack:output:0/instrumentalness/strided_slice/stack_1:output:0/instrumentalness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 instrumentalness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :®
instrumentalness/Reshape/shapePack'instrumentalness/strided_slice:output:0)instrumentalness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:§
instrumentalness/ReshapeReshape$instrumentalness/ExpandDims:output:0'instrumentalness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€]
key/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€w
key/ExpandDims
ExpandDims
features_6key/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:€€€€€€€€€j
key/CastCastkey/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:€€€€€€€€€E
	key/ShapeShapekey/Cast:y:0*
T0*
_output_shapes
:a
key/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
key/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
key/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:е
key/strided_sliceStridedSlicekey/Shape:output:0 key/strided_slice/stack:output:0"key/strided_slice/stack_1:output:0"key/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
key/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Б
key/Reshape/shapePackkey/strided_slice:output:0key/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
key/ReshapeReshapekey/Cast:y:0key/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€b
liveness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Б
liveness/ExpandDims
ExpandDims
features_7 liveness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€Z
liveness/ShapeShapeliveness/ExpandDims:output:0*
T0*
_output_shapes
:f
liveness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
liveness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
liveness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
liveness/strided_sliceStridedSliceliveness/Shape:output:0%liveness/strided_slice/stack:output:0'liveness/strided_slice/stack_1:output:0'liveness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
liveness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Р
liveness/Reshape/shapePackliveness/strided_slice:output:0!liveness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:М
liveness/ReshapeReshapeliveness/ExpandDims:output:0liveness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€b
loudness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Б
loudness/ExpandDims
ExpandDims
features_8 loudness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€Z
loudness/ShapeShapeloudness/ExpandDims:output:0*
T0*
_output_shapes
:f
loudness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
loudness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
loudness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
loudness/strided_sliceStridedSliceloudness/Shape:output:0%loudness/strided_slice/stack:output:0'loudness/strided_slice/stack_1:output:0'loudness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
loudness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Р
loudness/Reshape/shapePackloudness/strided_slice:output:0!loudness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:М
loudness/ReshapeReshapeloudness/ExpandDims:output:0loudness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€e
speechiness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€З
speechiness/ExpandDims
ExpandDims
features_9#speechiness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
speechiness/ShapeShapespeechiness/ExpandDims:output:0*
T0*
_output_shapes
:i
speechiness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!speechiness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!speechiness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
speechiness/strided_sliceStridedSlicespeechiness/Shape:output:0(speechiness/strided_slice/stack:output:0*speechiness/strided_slice/stack_1:output:0*speechiness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
speechiness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Щ
speechiness/Reshape/shapePack"speechiness/strided_slice:output:0$speechiness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Х
speechiness/ReshapeReshapespeechiness/ExpandDims:output:0"speechiness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€_
tempo/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€|
tempo/ExpandDims
ExpandDimsfeatures_10tempo/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€T
tempo/ShapeShapetempo/ExpandDims:output:0*
T0*
_output_shapes
:c
tempo/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
tempo/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
tempo/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
tempo/strided_sliceStridedSlicetempo/Shape:output:0"tempo/strided_slice/stack:output:0$tempo/strided_slice/stack_1:output:0$tempo/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
tempo/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :З
tempo/Reshape/shapePacktempo/strided_slice:output:0tempo/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Г
tempo/ReshapeReshapetempo/ExpandDims:output:0tempo/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€a
valence/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€А
valence/ExpandDims
ExpandDimsfeatures_11valence/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
valence/ShapeShapevalence/ExpandDims:output:0*
T0*
_output_shapes
:e
valence/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
valence/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
valence/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
valence/strided_sliceStridedSlicevalence/Shape:output:0$valence/strided_slice/stack:output:0&valence/strided_slice/stack_1:output:0&valence/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
valence/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Н
valence/Reshape/shapePackvalence/strided_slice:output:0 valence/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Й
valence/ReshapeReshapevalence/ExpandDims:output:0valence/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Ф
concatConcatV2acousticness/Reshape:output:0danceability/Reshape:output:0duration/Reshape:output:0energy/Reshape:output:0!instrumentalness/Reshape:output:0key/Reshape:output:0liveness/Reshape:output:0loudness/Reshape:output:0speechiness/Reshape:output:0tempo/Reshape:output:0valence/Reshape:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*…
_input_shapesЈ
і:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:M I
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
features:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
features:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
features:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
features:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
features:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
features:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
features:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
features:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
features:M	I
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
features:M
I
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
features:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
features
Џ
a
C__inference_dropout_layer_call_and_return_conditional_losses_989953

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:€€€€€€€€€А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:€€€€€€€€€А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ъ

ф
C__inference_dense_1_layer_call_and_return_conditional_losses_989858

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Р

b
C__inference_dropout_layer_call_and_return_conditional_losses_988416

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€АC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>І
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
√
Ц
(__inference_dense_7_layer_call_fn_989994

inputs
unknown:	А
	unknown_0:
identityИҐStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_988334o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ћ
Ѕ
+__inference_sequential_layer_call_fn_988851
id
acousticness
danceability
duration	

energy
instrumentalness
key	
liveness
loudness
speechiness	
tempo
valence
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:	@А
	unknown_6:	А
	unknown_7:
АА
	unknown_8:	А
	unknown_9:
АА

unknown_10:	А

unknown_11:
АА

unknown_12:	А

unknown_13:	А

unknown_14:
identityИҐStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallidacousticnessdanceabilitydurationenergyinstrumentalnesskeylivenessloudnessspeechinesstempovalenceunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*'
Tin 
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*2
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_988768o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*й
_input_shapes„
‘:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:G C
#
_output_shapes
:€€€€€€€€€

_user_specified_nameID:QM
#
_output_shapes
:€€€€€€€€€
&
_user_specified_nameacousticness:QM
#
_output_shapes
:€€€€€€€€€
&
_user_specified_namedanceability:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
duration:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameenergy:UQ
#
_output_shapes
:€€€€€€€€€
*
_user_specified_nameinstrumentalness:HD
#
_output_shapes
:€€€€€€€€€

_user_specified_namekey:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
liveness:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
loudness:P	L
#
_output_shapes
:€€€€€€€€€
%
_user_specified_namespeechiness:J
F
#
_output_shapes
:€€€€€€€€€

_user_specified_nametempo:LH
#
_output_shapes
:€€€€€€€€€
!
_user_specified_name	valence
Т
’
/__inference_dense_features_layer_call_fn_989544
features_id
features_acousticness
features_danceability
features_duration	
features_energy
features_instrumentalness
features_key	
features_liveness
features_loudness
features_speechiness
features_tempo
features_valence
identityЮ
PartitionedCallPartitionedCallfeatures_idfeatures_acousticnessfeatures_danceabilityfeatures_durationfeatures_energyfeatures_instrumentalnessfeatures_keyfeatures_livenessfeatures_loudnessfeatures_speechinessfeatures_tempofeatures_valence*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_988195`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*…
_input_shapesЈ
і:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:P L
#
_output_shapes
:€€€€€€€€€
%
_user_specified_namefeatures_id:ZV
#
_output_shapes
:€€€€€€€€€
/
_user_specified_namefeatures_acousticness:ZV
#
_output_shapes
:€€€€€€€€€
/
_user_specified_namefeatures_danceability:VR
#
_output_shapes
:€€€€€€€€€
+
_user_specified_namefeatures_duration:TP
#
_output_shapes
:€€€€€€€€€
)
_user_specified_namefeatures_energy:^Z
#
_output_shapes
:€€€€€€€€€
3
_user_specified_namefeatures_instrumentalness:QM
#
_output_shapes
:€€€€€€€€€
&
_user_specified_namefeatures_key:VR
#
_output_shapes
:€€€€€€€€€
+
_user_specified_namefeatures_liveness:VR
#
_output_shapes
:€€€€€€€€€
+
_user_specified_namefeatures_loudness:Y	U
#
_output_shapes
:€€€€€€€€€
.
_user_specified_namefeatures_speechiness:S
O
#
_output_shapes
:€€€€€€€€€
(
_user_specified_namefeatures_tempo:UQ
#
_output_shapes
:€€€€€€€€€
*
_user_specified_namefeatures_valence
ЈA
Щ
F__inference_sequential_layer_call_and_return_conditional_losses_988917
id
acousticness
danceability
duration	

energy
instrumentalness
key	
liveness
loudness
speechiness	
tempo
valence
dense_988875:@
dense_988877:@ 
dense_1_988880:@@
dense_1_988882:@ 
dense_2_988885:@@
dense_2_988887:@!
dense_3_988890:	@А
dense_3_988892:	А"
dense_4_988895:
АА
dense_4_988897:	А"
dense_5_988900:
АА
dense_5_988902:	А"
dense_6_988906:
АА
dense_6_988908:	А!
dense_7_988911:	А
dense_7_988913:
identityИҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCallҐdense_3/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallҐdense_5/StatefulPartitionedCallҐdense_6/StatefulPartitionedCallҐdense_7/StatefulPartitionedCallf
dense_features/CastCastacousticness*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€h
dense_features/Cast_1Castdanceability*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€b
dense_features/Cast_2Castenergy*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€l
dense_features/Cast_3Castinstrumentalness*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€d
dense_features/Cast_4Castliveness*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€d
dense_features/Cast_5Castloudness*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€g
dense_features/Cast_6Castspeechiness*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€a
dense_features/Cast_7Casttempo*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€c
dense_features/Cast_8Castvalence*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€Ћ
dense_features/PartitionedCallPartitionedCalliddense_features/Cast:y:0dense_features/Cast_1:y:0durationdense_features/Cast_2:y:0dense_features/Cast_3:y:0keydense_features/Cast_4:y:0dense_features/Cast_5:y:0dense_features/Cast_6:y:0dense_features/Cast_7:y:0dense_features/Cast_8:y:0*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_988195Е
dense/StatefulPartitionedCallStatefulPartitionedCall'dense_features/PartitionedCall:output:0dense_988875dense_988877*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_988208М
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_988880dense_1_988882*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_988225О
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_988885dense_2_988887*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_988242П
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_988890dense_3_988892*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_988259П
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_988895dense_4_988897*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_988276П
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_988900dense_5_988902*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_988293ў
dropout/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_988304З
dense_6/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_6_988906dense_6_988908*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_988317О
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_988911dense_7_988913*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_988334w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€‘
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*й
_input_shapes„
‘:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:G C
#
_output_shapes
:€€€€€€€€€

_user_specified_nameID:QM
#
_output_shapes
:€€€€€€€€€
&
_user_specified_nameacousticness:QM
#
_output_shapes
:€€€€€€€€€
&
_user_specified_namedanceability:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
duration:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameenergy:UQ
#
_output_shapes
:€€€€€€€€€
*
_user_specified_nameinstrumentalness:HD
#
_output_shapes
:€€€€€€€€€

_user_specified_namekey:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
liveness:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
loudness:P	L
#
_output_shapes
:€€€€€€€€€
%
_user_specified_namespeechiness:J
F
#
_output_shapes
:€€€€€€€€€

_user_specified_nametempo:LH
#
_output_shapes
:€€€€€€€€€
!
_user_specified_name	valence
ір
л
F__inference_sequential_layer_call_and_return_conditional_losses_989528
	inputs_id
inputs_acousticness
inputs_danceability
inputs_duration	
inputs_energy
inputs_instrumentalness

inputs_key	
inputs_liveness
inputs_loudness
inputs_speechiness
inputs_tempo
inputs_valence6
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@@5
'dense_1_biasadd_readvariableop_resource:@8
&dense_2_matmul_readvariableop_resource:@@5
'dense_2_biasadd_readvariableop_resource:@9
&dense_3_matmul_readvariableop_resource:	@А6
'dense_3_biasadd_readvariableop_resource:	А:
&dense_4_matmul_readvariableop_resource:
АА6
'dense_4_biasadd_readvariableop_resource:	А:
&dense_5_matmul_readvariableop_resource:
АА6
'dense_5_biasadd_readvariableop_resource:	А:
&dense_6_matmul_readvariableop_resource:
АА6
'dense_6_biasadd_readvariableop_resource:	А9
&dense_7_matmul_readvariableop_resource:	А5
'dense_7_biasadd_readvariableop_resource:
identityИҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐdense_2/BiasAdd/ReadVariableOpҐdense_2/MatMul/ReadVariableOpҐdense_3/BiasAdd/ReadVariableOpҐdense_3/MatMul/ReadVariableOpҐdense_4/BiasAdd/ReadVariableOpҐdense_4/MatMul/ReadVariableOpҐdense_5/BiasAdd/ReadVariableOpҐdense_5/MatMul/ReadVariableOpҐdense_6/BiasAdd/ReadVariableOpҐdense_6/MatMul/ReadVariableOpҐdense_7/BiasAdd/ReadVariableOpҐdense_7/MatMul/ReadVariableOpm
dense_features/CastCastinputs_acousticness*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€o
dense_features/Cast_1Castinputs_danceability*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€i
dense_features/Cast_2Castinputs_energy*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€s
dense_features/Cast_3Castinputs_instrumentalness*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€k
dense_features/Cast_4Castinputs_liveness*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€k
dense_features/Cast_5Castinputs_loudness*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€n
dense_features/Cast_6Castinputs_speechiness*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€h
dense_features/Cast_7Castinputs_tempo*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€j
dense_features/Cast_8Castinputs_valence*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€u
*dense_features/acousticness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
&dense_features/acousticness/ExpandDims
ExpandDimsdense_features/Cast:y:03dense_features/acousticness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€А
!dense_features/acousticness/ShapeShape/dense_features/acousticness/ExpandDims:output:0*
T0*
_output_shapes
:y
/dense_features/acousticness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_features/acousticness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_features/acousticness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ё
)dense_features/acousticness/strided_sliceStridedSlice*dense_features/acousticness/Shape:output:08dense_features/acousticness/strided_slice/stack:output:0:dense_features/acousticness/strided_slice/stack_1:output:0:dense_features/acousticness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+dense_features/acousticness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :…
)dense_features/acousticness/Reshape/shapePack2dense_features/acousticness/strided_slice:output:04dense_features/acousticness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:≈
#dense_features/acousticness/ReshapeReshape/dense_features/acousticness/ExpandDims:output:02dense_features/acousticness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€u
*dense_features/danceability/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ґ
&dense_features/danceability/ExpandDims
ExpandDimsdense_features/Cast_1:y:03dense_features/danceability/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€А
!dense_features/danceability/ShapeShape/dense_features/danceability/ExpandDims:output:0*
T0*
_output_shapes
:y
/dense_features/danceability/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_features/danceability/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_features/danceability/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ё
)dense_features/danceability/strided_sliceStridedSlice*dense_features/danceability/Shape:output:08dense_features/danceability/strided_slice/stack:output:0:dense_features/danceability/strided_slice/stack_1:output:0:dense_features/danceability/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+dense_features/danceability/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :…
)dense_features/danceability/Reshape/shapePack2dense_features/danceability/strided_slice:output:04dense_features/danceability/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:≈
#dense_features/danceability/ReshapeReshape/dense_features/danceability/ExpandDims:output:02dense_features/danceability/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€q
&dense_features/duration/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€§
"dense_features/duration/ExpandDims
ExpandDimsinputs_duration/dense_features/duration/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:€€€€€€€€€Т
dense_features/duration/CastCast+dense_features/duration/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:€€€€€€€€€m
dense_features/duration/ShapeShape dense_features/duration/Cast:y:0*
T0*
_output_shapes
:u
+dense_features/duration/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-dense_features/duration/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-dense_features/duration/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:…
%dense_features/duration/strided_sliceStridedSlice&dense_features/duration/Shape:output:04dense_features/duration/strided_slice/stack:output:06dense_features/duration/strided_slice/stack_1:output:06dense_features/duration/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'dense_features/duration/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :љ
%dense_features/duration/Reshape/shapePack.dense_features/duration/strided_slice:output:00dense_features/duration/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ѓ
dense_features/duration/ReshapeReshape dense_features/duration/Cast:y:0.dense_features/duration/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€o
$dense_features/energy/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€™
 dense_features/energy/ExpandDims
ExpandDimsdense_features/Cast_2:y:0-dense_features/energy/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€t
dense_features/energy/ShapeShape)dense_features/energy/ExpandDims:output:0*
T0*
_output_shapes
:s
)dense_features/energy/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+dense_features/energy/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+dense_features/energy/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:њ
#dense_features/energy/strided_sliceStridedSlice$dense_features/energy/Shape:output:02dense_features/energy/strided_slice/stack:output:04dense_features/energy/strided_slice/stack_1:output:04dense_features/energy/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%dense_features/energy/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ј
#dense_features/energy/Reshape/shapePack,dense_features/energy/strided_slice:output:0.dense_features/energy/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:≥
dense_features/energy/ReshapeReshape)dense_features/energy/ExpandDims:output:0,dense_features/energy/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€y
.dense_features/instrumentalness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Њ
*dense_features/instrumentalness/ExpandDims
ExpandDimsdense_features/Cast_3:y:07dense_features/instrumentalness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€И
%dense_features/instrumentalness/ShapeShape3dense_features/instrumentalness/ExpandDims:output:0*
T0*
_output_shapes
:}
3dense_features/instrumentalness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5dense_features/instrumentalness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5dense_features/instrumentalness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:с
-dense_features/instrumentalness/strided_sliceStridedSlice.dense_features/instrumentalness/Shape:output:0<dense_features/instrumentalness/strided_slice/stack:output:0>dense_features/instrumentalness/strided_slice/stack_1:output:0>dense_features/instrumentalness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
/dense_features/instrumentalness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :’
-dense_features/instrumentalness/Reshape/shapePack6dense_features/instrumentalness/strided_slice:output:08dense_features/instrumentalness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:—
'dense_features/instrumentalness/ReshapeReshape3dense_features/instrumentalness/ExpandDims:output:06dense_features/instrumentalness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€l
!dense_features/key/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Х
dense_features/key/ExpandDims
ExpandDims
inputs_key*dense_features/key/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:€€€€€€€€€И
dense_features/key/CastCast&dense_features/key/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:€€€€€€€€€c
dense_features/key/ShapeShapedense_features/key/Cast:y:0*
T0*
_output_shapes
:p
&dense_features/key/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/key/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/key/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∞
 dense_features/key/strided_sliceStridedSlice!dense_features/key/Shape:output:0/dense_features/key/strided_slice/stack:output:01dense_features/key/strided_slice/stack_1:output:01dense_features/key/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/key/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ѓ
 dense_features/key/Reshape/shapePack)dense_features/key/strided_slice:output:0+dense_features/key/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Я
dense_features/key/ReshapeReshapedense_features/key/Cast:y:0)dense_features/key/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€q
&dense_features/liveness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Ѓ
"dense_features/liveness/ExpandDims
ExpandDimsdense_features/Cast_4:y:0/dense_features/liveness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€x
dense_features/liveness/ShapeShape+dense_features/liveness/ExpandDims:output:0*
T0*
_output_shapes
:u
+dense_features/liveness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-dense_features/liveness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-dense_features/liveness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:…
%dense_features/liveness/strided_sliceStridedSlice&dense_features/liveness/Shape:output:04dense_features/liveness/strided_slice/stack:output:06dense_features/liveness/strided_slice/stack_1:output:06dense_features/liveness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'dense_features/liveness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :љ
%dense_features/liveness/Reshape/shapePack.dense_features/liveness/strided_slice:output:00dense_features/liveness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:є
dense_features/liveness/ReshapeReshape+dense_features/liveness/ExpandDims:output:0.dense_features/liveness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€q
&dense_features/loudness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Ѓ
"dense_features/loudness/ExpandDims
ExpandDimsdense_features/Cast_5:y:0/dense_features/loudness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€x
dense_features/loudness/ShapeShape+dense_features/loudness/ExpandDims:output:0*
T0*
_output_shapes
:u
+dense_features/loudness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-dense_features/loudness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-dense_features/loudness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:…
%dense_features/loudness/strided_sliceStridedSlice&dense_features/loudness/Shape:output:04dense_features/loudness/strided_slice/stack:output:06dense_features/loudness/strided_slice/stack_1:output:06dense_features/loudness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'dense_features/loudness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :љ
%dense_features/loudness/Reshape/shapePack.dense_features/loudness/strided_slice:output:00dense_features/loudness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:є
dense_features/loudness/ReshapeReshape+dense_features/loudness/ExpandDims:output:0.dense_features/loudness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€t
)dense_features/speechiness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
%dense_features/speechiness/ExpandDims
ExpandDimsdense_features/Cast_6:y:02dense_features/speechiness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€~
 dense_features/speechiness/ShapeShape.dense_features/speechiness/ExpandDims:output:0*
T0*
_output_shapes
:x
.dense_features/speechiness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0dense_features/speechiness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0dense_features/speechiness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ў
(dense_features/speechiness/strided_sliceStridedSlice)dense_features/speechiness/Shape:output:07dense_features/speechiness/strided_slice/stack:output:09dense_features/speechiness/strided_slice/stack_1:output:09dense_features/speechiness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*dense_features/speechiness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :∆
(dense_features/speechiness/Reshape/shapePack1dense_features/speechiness/strided_slice:output:03dense_features/speechiness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:¬
"dense_features/speechiness/ReshapeReshape.dense_features/speechiness/ExpandDims:output:01dense_features/speechiness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€n
#dense_features/tempo/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€®
dense_features/tempo/ExpandDims
ExpandDimsdense_features/Cast_7:y:0,dense_features/tempo/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€r
dense_features/tempo/ShapeShape(dense_features/tempo/ExpandDims:output:0*
T0*
_output_shapes
:r
(dense_features/tempo/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*dense_features/tempo/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*dense_features/tempo/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"dense_features/tempo/strided_sliceStridedSlice#dense_features/tempo/Shape:output:01dense_features/tempo/strided_slice/stack:output:03dense_features/tempo/strided_slice/stack_1:output:03dense_features/tempo/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$dense_features/tempo/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :і
"dense_features/tempo/Reshape/shapePack+dense_features/tempo/strided_slice:output:0-dense_features/tempo/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:∞
dense_features/tempo/ReshapeReshape(dense_features/tempo/ExpandDims:output:0+dense_features/tempo/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€p
%dense_features/valence/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ђ
!dense_features/valence/ExpandDims
ExpandDimsdense_features/Cast_8:y:0.dense_features/valence/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€v
dense_features/valence/ShapeShape*dense_features/valence/ExpandDims:output:0*
T0*
_output_shapes
:t
*dense_features/valence/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,dense_features/valence/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,dense_features/valence/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ƒ
$dense_features/valence/strided_sliceStridedSlice%dense_features/valence/Shape:output:03dense_features/valence/strided_slice/stack:output:05dense_features/valence/strided_slice/stack_1:output:05dense_features/valence/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&dense_features/valence/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ї
$dense_features/valence/Reshape/shapePack-dense_features/valence/strided_slice:output:0/dense_features/valence/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:ґ
dense_features/valence/ReshapeReshape*dense_features/valence/ExpandDims:output:0-dense_features/valence/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€e
dense_features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€„
dense_features/concatConcatV2,dense_features/acousticness/Reshape:output:0,dense_features/danceability/Reshape:output:0(dense_features/duration/Reshape:output:0&dense_features/energy/Reshape:output:00dense_features/instrumentalness/Reshape:output:0#dense_features/key/Reshape:output:0(dense_features/liveness/Reshape:output:0(dense_features/loudness/Reshape:output:0+dense_features/speechiness/Reshape:output:0%dense_features/tempo/Reshape:output:0'dense_features/valence/Reshape:output:0#dense_features/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€А
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Н
dense/MatMulMatMuldense_features/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0И
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Д
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0Л
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Д
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0Н
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@В
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0О
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Е
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype0О
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АГ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0П
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аa
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€АЖ
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0О
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АГ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0П
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аa
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€АЖ
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0О
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АГ
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0П
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аa
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€АZ
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?Й
dropout/dropout/MulMuldense_5/Relu:activations:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А_
dropout/dropout/ShapeShapedense_5/Relu:activations:0*
T0*
_output_shapes
:Э
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>њ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€А\
dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    і
dropout/dropout/SelectV2SelectV2 dropout/dropout/GreaterEqual:z:0dropout/dropout/Mul:z:0 dropout/dropout/Const_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€АЖ
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Х
dense_6/MatMulMatMul!dropout/dropout/SelectV2:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АГ
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0П
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аa
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€АЕ
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Н
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€f
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€h
IdentityIdentitydense_7/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*й
_input_shapes„
‘:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:N J
#
_output_shapes
:€€€€€€€€€
#
_user_specified_name	inputs_id:XT
#
_output_shapes
:€€€€€€€€€
-
_user_specified_nameinputs_acousticness:XT
#
_output_shapes
:€€€€€€€€€
-
_user_specified_nameinputs_danceability:TP
#
_output_shapes
:€€€€€€€€€
)
_user_specified_nameinputs_duration:RN
#
_output_shapes
:€€€€€€€€€
'
_user_specified_nameinputs_energy:\X
#
_output_shapes
:€€€€€€€€€
1
_user_specified_nameinputs_instrumentalness:OK
#
_output_shapes
:€€€€€€€€€
$
_user_specified_name
inputs_key:TP
#
_output_shapes
:€€€€€€€€€
)
_user_specified_nameinputs_liveness:TP
#
_output_shapes
:€€€€€€€€€
)
_user_specified_nameinputs_loudness:W	S
#
_output_shapes
:€€€€€€€€€
,
_user_specified_nameinputs_speechiness:Q
M
#
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs_tempo:SO
#
_output_shapes
:€€€€€€€€€
(
_user_specified_nameinputs_valence
¶

ч
C__inference_dense_5_layer_call_and_return_conditional_losses_989938

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
£

х
C__inference_dense_7_layer_call_and_return_conditional_losses_988334

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Р

b
C__inference_dropout_layer_call_and_return_conditional_losses_989965

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  †?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€АC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЌћL>І
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:€€€€€€€€€АT
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Ф
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аb
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
єB
є
F__inference_sequential_layer_call_and_return_conditional_losses_988768

inputs
inputs_1
inputs_2
inputs_3	
inputs_4
inputs_5
inputs_6	
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
dense_988726:@
dense_988728:@ 
dense_1_988731:@@
dense_1_988733:@ 
dense_2_988736:@@
dense_2_988738:@!
dense_3_988741:	@А
dense_3_988743:	А"
dense_4_988746:
АА
dense_4_988748:	А"
dense_5_988751:
АА
dense_5_988753:	А"
dense_6_988757:
АА
dense_6_988759:	А!
dense_7_988762:	А
dense_7_988764:
identityИҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCallҐdense_3/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallҐdense_5/StatefulPartitionedCallҐdense_6/StatefulPartitionedCallҐdense_7/StatefulPartitionedCallҐdropout/StatefulPartitionedCallb
dense_features/CastCastinputs_1*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€d
dense_features/Cast_1Castinputs_2*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€d
dense_features/Cast_2Castinputs_4*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€d
dense_features/Cast_3Castinputs_5*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€d
dense_features/Cast_4Castinputs_7*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€d
dense_features/Cast_5Castinputs_8*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€d
dense_features/Cast_6Castinputs_9*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€e
dense_features/Cast_7Cast	inputs_10*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€e
dense_features/Cast_8Cast	inputs_11*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€‘
dense_features/PartitionedCallPartitionedCallinputsdense_features/Cast:y:0dense_features/Cast_1:y:0inputs_3dense_features/Cast_2:y:0dense_features/Cast_3:y:0inputs_6dense_features/Cast_4:y:0dense_features/Cast_5:y:0dense_features/Cast_6:y:0dense_features/Cast_7:y:0dense_features/Cast_8:y:0*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_988638Е
dense/StatefulPartitionedCallStatefulPartitionedCall'dense_features/PartitionedCall:output:0dense_988726dense_988728*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_988208М
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_988731dense_1_988733*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_988225О
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_988736dense_2_988738*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_988242П
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_988741dense_3_988743*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_988259П
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_988746dense_4_988748*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_988276П
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_988751dense_5_988753*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_988293й
dropout/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_988416П
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_6_988757dense_6_988759*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_988317О
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_988762dense_7_988764*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_988334w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ц
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*й
_input_shapes„
‘:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:K G
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:K	G
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:K
G
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Я
D
(__inference_dropout_layer_call_fn_989943

inputs
identityѓ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_988304a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:€€€€€€€€€А:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ъ

ф
C__inference_dense_2_layer_call_and_return_conditional_losses_989878

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ј
Х
(__inference_dense_2_layer_call_fn_989867

inputs
unknown:@@
	unknown_0:@
identityИҐStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_988242o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
«
Ш
(__inference_dense_6_layer_call_fn_989974

inputs
unknown:
АА
	unknown_0:	А
identityИҐStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_988317p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
»
Х
+__inference_sequential_layer_call_fn_989131
	inputs_id
inputs_acousticness
inputs_danceability
inputs_duration	
inputs_energy
inputs_instrumentalness

inputs_key	
inputs_liveness
inputs_loudness
inputs_speechiness
inputs_tempo
inputs_valence
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:	@А
	unknown_6:	А
	unknown_7:
АА
	unknown_8:	А
	unknown_9:
АА

unknown_10:	А

unknown_11:
АА

unknown_12:	А

unknown_13:	А

unknown_14:
identityИҐStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCall	inputs_idinputs_acousticnessinputs_danceabilityinputs_durationinputs_energyinputs_instrumentalness
inputs_keyinputs_livenessinputs_loudnessinputs_speechinessinputs_tempoinputs_valenceunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*'
Tin 
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*2
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_988768o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*й
_input_shapes„
‘:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
#
_output_shapes
:€€€€€€€€€
#
_user_specified_name	inputs_id:XT
#
_output_shapes
:€€€€€€€€€
-
_user_specified_nameinputs_acousticness:XT
#
_output_shapes
:€€€€€€€€€
-
_user_specified_nameinputs_danceability:TP
#
_output_shapes
:€€€€€€€€€
)
_user_specified_nameinputs_duration:RN
#
_output_shapes
:€€€€€€€€€
'
_user_specified_nameinputs_energy:\X
#
_output_shapes
:€€€€€€€€€
1
_user_specified_nameinputs_instrumentalness:OK
#
_output_shapes
:€€€€€€€€€
$
_user_specified_name
inputs_key:TP
#
_output_shapes
:€€€€€€€€€
)
_user_specified_nameinputs_liveness:TP
#
_output_shapes
:€€€€€€€€€
)
_user_specified_nameinputs_loudness:W	S
#
_output_shapes
:€€€€€€€€€
,
_user_specified_nameinputs_speechiness:Q
M
#
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs_tempo:SO
#
_output_shapes
:€€€€€€€€€
(
_user_specified_nameinputs_valence
ƒ
Ч
(__inference_dense_3_layer_call_fn_989887

inputs
unknown:	@А
	unknown_0:	А
identityИҐStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_988259p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
ЩA
Ч
F__inference_sequential_layer_call_and_return_conditional_losses_988341

inputs
inputs_1
inputs_2
inputs_3	
inputs_4
inputs_5
inputs_6	
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
dense_988209:@
dense_988211:@ 
dense_1_988226:@@
dense_1_988228:@ 
dense_2_988243:@@
dense_2_988245:@!
dense_3_988260:	@А
dense_3_988262:	А"
dense_4_988277:
АА
dense_4_988279:	А"
dense_5_988294:
АА
dense_5_988296:	А"
dense_6_988318:
АА
dense_6_988320:	А!
dense_7_988335:	А
dense_7_988337:
identityИҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCallҐdense_3/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallҐdense_5/StatefulPartitionedCallҐdense_6/StatefulPartitionedCallҐdense_7/StatefulPartitionedCallb
dense_features/CastCastinputs_1*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€d
dense_features/Cast_1Castinputs_2*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€d
dense_features/Cast_2Castinputs_4*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€d
dense_features/Cast_3Castinputs_5*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€d
dense_features/Cast_4Castinputs_7*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€d
dense_features/Cast_5Castinputs_8*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€d
dense_features/Cast_6Castinputs_9*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€e
dense_features/Cast_7Cast	inputs_10*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€e
dense_features/Cast_8Cast	inputs_11*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€‘
dense_features/PartitionedCallPartitionedCallinputsdense_features/Cast:y:0dense_features/Cast_1:y:0inputs_3dense_features/Cast_2:y:0dense_features/Cast_3:y:0inputs_6dense_features/Cast_4:y:0dense_features/Cast_5:y:0dense_features/Cast_6:y:0dense_features/Cast_7:y:0dense_features/Cast_8:y:0*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_988195Е
dense/StatefulPartitionedCallStatefulPartitionedCall'dense_features/PartitionedCall:output:0dense_988209dense_988211*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_988208М
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_988226dense_1_988228*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_988225О
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_988243dense_2_988245*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_988242П
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_988260dense_3_988262*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_988259П
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_988277dense_4_988279*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_988276П
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_988294dense_5_988296*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_988293ў
dropout/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_988304З
dense_6/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_6_988318dense_6_988320*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_988317О
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_988335dense_7_988337*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_988334w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€‘
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*й
_input_shapes„
‘:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:K G
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:K	G
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:K
G
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ш

т
A__inference_dense_layer_call_and_return_conditional_losses_989838

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ґ

ц
C__inference_dense_3_layer_call_and_return_conditional_losses_989898

inputs1
matmul_readvariableop_resource:	@А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
чА
Ъ
J__inference_dense_features_layer_call_and_return_conditional_losses_988195
features

features_1

features_2

features_3	

features_4

features_5

features_6	

features_7

features_8

features_9
features_10
features_11
identityf
acousticness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Й
acousticness/ExpandDims
ExpandDims
features_1$acousticness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€b
acousticness/ShapeShape acousticness/ExpandDims:output:0*
T0*
_output_shapes
:j
 acousticness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"acousticness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"acousticness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
acousticness/strided_sliceStridedSliceacousticness/Shape:output:0)acousticness/strided_slice/stack:output:0+acousticness/strided_slice/stack_1:output:0+acousticness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
acousticness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ь
acousticness/Reshape/shapePack#acousticness/strided_slice:output:0%acousticness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ш
acousticness/ReshapeReshape acousticness/ExpandDims:output:0#acousticness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€f
danceability/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Й
danceability/ExpandDims
ExpandDims
features_2$danceability/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€b
danceability/ShapeShape danceability/ExpandDims:output:0*
T0*
_output_shapes
:j
 danceability/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"danceability/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"danceability/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
danceability/strided_sliceStridedSlicedanceability/Shape:output:0)danceability/strided_slice/stack:output:0+danceability/strided_slice/stack_1:output:0+danceability/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
danceability/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ь
danceability/Reshape/shapePack#danceability/strided_slice:output:0%danceability/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ш
danceability/ReshapeReshape danceability/ExpandDims:output:0#danceability/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€b
duration/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Б
duration/ExpandDims
ExpandDims
features_3 duration/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:€€€€€€€€€t
duration/CastCastduration/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:€€€€€€€€€O
duration/ShapeShapeduration/Cast:y:0*
T0*
_output_shapes
:f
duration/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
duration/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
duration/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
duration/strided_sliceStridedSliceduration/Shape:output:0%duration/strided_slice/stack:output:0'duration/strided_slice/stack_1:output:0'duration/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
duration/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Р
duration/Reshape/shapePackduration/strided_slice:output:0!duration/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Б
duration/ReshapeReshapeduration/Cast:y:0duration/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
energy/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€}
energy/ExpandDims
ExpandDims
features_4energy/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
energy/ShapeShapeenergy/ExpandDims:output:0*
T0*
_output_shapes
:d
energy/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
energy/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
energy/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ф
energy/strided_sliceStridedSliceenergy/Shape:output:0#energy/strided_slice/stack:output:0%energy/strided_slice/stack_1:output:0%energy/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
energy/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :К
energy/Reshape/shapePackenergy/strided_slice:output:0energy/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ж
energy/ReshapeReshapeenergy/ExpandDims:output:0energy/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€j
instrumentalness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€С
instrumentalness/ExpandDims
ExpandDims
features_5(instrumentalness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€j
instrumentalness/ShapeShape$instrumentalness/ExpandDims:output:0*
T0*
_output_shapes
:n
$instrumentalness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&instrumentalness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&instrumentalness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¶
instrumentalness/strided_sliceStridedSliceinstrumentalness/Shape:output:0-instrumentalness/strided_slice/stack:output:0/instrumentalness/strided_slice/stack_1:output:0/instrumentalness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 instrumentalness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :®
instrumentalness/Reshape/shapePack'instrumentalness/strided_slice:output:0)instrumentalness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:§
instrumentalness/ReshapeReshape$instrumentalness/ExpandDims:output:0'instrumentalness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€]
key/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€w
key/ExpandDims
ExpandDims
features_6key/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:€€€€€€€€€j
key/CastCastkey/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:€€€€€€€€€E
	key/ShapeShapekey/Cast:y:0*
T0*
_output_shapes
:a
key/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
key/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
key/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:е
key/strided_sliceStridedSlicekey/Shape:output:0 key/strided_slice/stack:output:0"key/strided_slice/stack_1:output:0"key/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
key/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Б
key/Reshape/shapePackkey/strided_slice:output:0key/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
key/ReshapeReshapekey/Cast:y:0key/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€b
liveness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Б
liveness/ExpandDims
ExpandDims
features_7 liveness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€Z
liveness/ShapeShapeliveness/ExpandDims:output:0*
T0*
_output_shapes
:f
liveness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
liveness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
liveness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
liveness/strided_sliceStridedSliceliveness/Shape:output:0%liveness/strided_slice/stack:output:0'liveness/strided_slice/stack_1:output:0'liveness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
liveness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Р
liveness/Reshape/shapePackliveness/strided_slice:output:0!liveness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:М
liveness/ReshapeReshapeliveness/ExpandDims:output:0liveness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€b
loudness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Б
loudness/ExpandDims
ExpandDims
features_8 loudness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€Z
loudness/ShapeShapeloudness/ExpandDims:output:0*
T0*
_output_shapes
:f
loudness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
loudness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
loudness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
loudness/strided_sliceStridedSliceloudness/Shape:output:0%loudness/strided_slice/stack:output:0'loudness/strided_slice/stack_1:output:0'loudness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
loudness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Р
loudness/Reshape/shapePackloudness/strided_slice:output:0!loudness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:М
loudness/ReshapeReshapeloudness/ExpandDims:output:0loudness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€e
speechiness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€З
speechiness/ExpandDims
ExpandDims
features_9#speechiness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
speechiness/ShapeShapespeechiness/ExpandDims:output:0*
T0*
_output_shapes
:i
speechiness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!speechiness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!speechiness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
speechiness/strided_sliceStridedSlicespeechiness/Shape:output:0(speechiness/strided_slice/stack:output:0*speechiness/strided_slice/stack_1:output:0*speechiness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
speechiness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Щ
speechiness/Reshape/shapePack"speechiness/strided_slice:output:0$speechiness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Х
speechiness/ReshapeReshapespeechiness/ExpandDims:output:0"speechiness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€_
tempo/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€|
tempo/ExpandDims
ExpandDimsfeatures_10tempo/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€T
tempo/ShapeShapetempo/ExpandDims:output:0*
T0*
_output_shapes
:c
tempo/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
tempo/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
tempo/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
tempo/strided_sliceStridedSlicetempo/Shape:output:0"tempo/strided_slice/stack:output:0$tempo/strided_slice/stack_1:output:0$tempo/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
tempo/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :З
tempo/Reshape/shapePacktempo/strided_slice:output:0tempo/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Г
tempo/ReshapeReshapetempo/ExpandDims:output:0tempo/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€a
valence/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€А
valence/ExpandDims
ExpandDimsfeatures_11valence/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
valence/ShapeShapevalence/ExpandDims:output:0*
T0*
_output_shapes
:e
valence/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
valence/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
valence/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
valence/strided_sliceStridedSlicevalence/Shape:output:0$valence/strided_slice/stack:output:0&valence/strided_slice/stack_1:output:0&valence/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
valence/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Н
valence/Reshape/shapePackvalence/strided_slice:output:0 valence/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Й
valence/ReshapeReshapevalence/ExpandDims:output:0valence/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Ф
concatConcatV2acousticness/Reshape:output:0danceability/Reshape:output:0duration/Reshape:output:0energy/Reshape:output:0!instrumentalness/Reshape:output:0key/Reshape:output:0liveness/Reshape:output:0loudness/Reshape:output:0speechiness/Reshape:output:0tempo/Reshape:output:0valence/Reshape:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*…
_input_shapesЈ
і:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:M I
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
features:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
features:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
features:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
features:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
features:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
features:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
features:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
features:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
features:M	I
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
features:M
I
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
features:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
features
Т
’
/__inference_dense_features_layer_call_fn_989560
features_id
features_acousticness
features_danceability
features_duration	
features_energy
features_instrumentalness
features_key	
features_liveness
features_loudness
features_speechiness
features_tempo
features_valence
identityЮ
PartitionedCallPartitionedCallfeatures_idfeatures_acousticnessfeatures_danceabilityfeatures_durationfeatures_energyfeatures_instrumentalnessfeatures_keyfeatures_livenessfeatures_loudnessfeatures_speechinessfeatures_tempofeatures_valence*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_988638`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*…
_input_shapesЈ
і:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:P L
#
_output_shapes
:€€€€€€€€€
%
_user_specified_namefeatures_id:ZV
#
_output_shapes
:€€€€€€€€€
/
_user_specified_namefeatures_acousticness:ZV
#
_output_shapes
:€€€€€€€€€
/
_user_specified_namefeatures_danceability:VR
#
_output_shapes
:€€€€€€€€€
+
_user_specified_namefeatures_duration:TP
#
_output_shapes
:€€€€€€€€€
)
_user_specified_namefeatures_energy:^Z
#
_output_shapes
:€€€€€€€€€
3
_user_specified_namefeatures_instrumentalness:QM
#
_output_shapes
:€€€€€€€€€
&
_user_specified_namefeatures_key:VR
#
_output_shapes
:€€€€€€€€€
+
_user_specified_namefeatures_liveness:VR
#
_output_shapes
:€€€€€€€€€
+
_user_specified_namefeatures_loudness:Y	U
#
_output_shapes
:€€€€€€€€€
.
_user_specified_namefeatures_speechiness:S
O
#
_output_shapes
:€€€€€€€€€
(
_user_specified_namefeatures_tempo:UQ
#
_output_shapes
:€€€€€€€€€
*
_user_specified_namefeatures_valence
¶

ч
C__inference_dense_6_layer_call_and_return_conditional_losses_989985

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
«
Ш
(__inference_dense_5_layer_call_fn_989927

inputs
unknown:
АА
	unknown_0:	А
identityИҐStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_988293p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ПГ
р
J__inference_dense_features_layer_call_and_return_conditional_losses_989818
features_id
features_acousticness
features_danceability
features_duration	
features_energy
features_instrumentalness
features_key	
features_liveness
features_loudness
features_speechiness
features_tempo
features_valence
identityf
acousticness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Ф
acousticness/ExpandDims
ExpandDimsfeatures_acousticness$acousticness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€b
acousticness/ShapeShape acousticness/ExpandDims:output:0*
T0*
_output_shapes
:j
 acousticness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"acousticness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"acousticness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
acousticness/strided_sliceStridedSliceacousticness/Shape:output:0)acousticness/strided_slice/stack:output:0+acousticness/strided_slice/stack_1:output:0+acousticness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
acousticness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ь
acousticness/Reshape/shapePack#acousticness/strided_slice:output:0%acousticness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ш
acousticness/ReshapeReshape acousticness/ExpandDims:output:0#acousticness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€f
danceability/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Ф
danceability/ExpandDims
ExpandDimsfeatures_danceability$danceability/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€b
danceability/ShapeShape danceability/ExpandDims:output:0*
T0*
_output_shapes
:j
 danceability/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"danceability/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"danceability/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
danceability/strided_sliceStridedSlicedanceability/Shape:output:0)danceability/strided_slice/stack:output:0+danceability/strided_slice/stack_1:output:0+danceability/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
danceability/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ь
danceability/Reshape/shapePack#danceability/strided_slice:output:0%danceability/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ш
danceability/ReshapeReshape danceability/ExpandDims:output:0#danceability/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€b
duration/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€И
duration/ExpandDims
ExpandDimsfeatures_duration duration/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:€€€€€€€€€t
duration/CastCastduration/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:€€€€€€€€€O
duration/ShapeShapeduration/Cast:y:0*
T0*
_output_shapes
:f
duration/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
duration/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
duration/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
duration/strided_sliceStridedSliceduration/Shape:output:0%duration/strided_slice/stack:output:0'duration/strided_slice/stack_1:output:0'duration/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
duration/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Р
duration/Reshape/shapePackduration/strided_slice:output:0!duration/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Б
duration/ReshapeReshapeduration/Cast:y:0duration/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
energy/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€В
energy/ExpandDims
ExpandDimsfeatures_energyenergy/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
energy/ShapeShapeenergy/ExpandDims:output:0*
T0*
_output_shapes
:d
energy/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
energy/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
energy/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ф
energy/strided_sliceStridedSliceenergy/Shape:output:0#energy/strided_slice/stack:output:0%energy/strided_slice/stack_1:output:0%energy/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
energy/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :К
energy/Reshape/shapePackenergy/strided_slice:output:0energy/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ж
energy/ReshapeReshapeenergy/ExpandDims:output:0energy/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€j
instrumentalness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€†
instrumentalness/ExpandDims
ExpandDimsfeatures_instrumentalness(instrumentalness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€j
instrumentalness/ShapeShape$instrumentalness/ExpandDims:output:0*
T0*
_output_shapes
:n
$instrumentalness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&instrumentalness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&instrumentalness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¶
instrumentalness/strided_sliceStridedSliceinstrumentalness/Shape:output:0-instrumentalness/strided_slice/stack:output:0/instrumentalness/strided_slice/stack_1:output:0/instrumentalness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 instrumentalness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :®
instrumentalness/Reshape/shapePack'instrumentalness/strided_slice:output:0)instrumentalness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:§
instrumentalness/ReshapeReshape$instrumentalness/ExpandDims:output:0'instrumentalness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€]
key/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€y
key/ExpandDims
ExpandDimsfeatures_keykey/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:€€€€€€€€€j
key/CastCastkey/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:€€€€€€€€€E
	key/ShapeShapekey/Cast:y:0*
T0*
_output_shapes
:a
key/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
key/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
key/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:е
key/strided_sliceStridedSlicekey/Shape:output:0 key/strided_slice/stack:output:0"key/strided_slice/stack_1:output:0"key/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
key/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Б
key/Reshape/shapePackkey/strided_slice:output:0key/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
key/ReshapeReshapekey/Cast:y:0key/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€b
liveness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€И
liveness/ExpandDims
ExpandDimsfeatures_liveness liveness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€Z
liveness/ShapeShapeliveness/ExpandDims:output:0*
T0*
_output_shapes
:f
liveness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
liveness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
liveness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
liveness/strided_sliceStridedSliceliveness/Shape:output:0%liveness/strided_slice/stack:output:0'liveness/strided_slice/stack_1:output:0'liveness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
liveness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Р
liveness/Reshape/shapePackliveness/strided_slice:output:0!liveness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:М
liveness/ReshapeReshapeliveness/ExpandDims:output:0liveness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€b
loudness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€И
loudness/ExpandDims
ExpandDimsfeatures_loudness loudness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€Z
loudness/ShapeShapeloudness/ExpandDims:output:0*
T0*
_output_shapes
:f
loudness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
loudness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
loudness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
loudness/strided_sliceStridedSliceloudness/Shape:output:0%loudness/strided_slice/stack:output:0'loudness/strided_slice/stack_1:output:0'loudness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
loudness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Р
loudness/Reshape/shapePackloudness/strided_slice:output:0!loudness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:М
loudness/ReshapeReshapeloudness/ExpandDims:output:0loudness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€e
speechiness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€С
speechiness/ExpandDims
ExpandDimsfeatures_speechiness#speechiness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
speechiness/ShapeShapespeechiness/ExpandDims:output:0*
T0*
_output_shapes
:i
speechiness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!speechiness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!speechiness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
speechiness/strided_sliceStridedSlicespeechiness/Shape:output:0(speechiness/strided_slice/stack:output:0*speechiness/strided_slice/stack_1:output:0*speechiness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
speechiness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Щ
speechiness/Reshape/shapePack"speechiness/strided_slice:output:0$speechiness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Х
speechiness/ReshapeReshapespeechiness/ExpandDims:output:0"speechiness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€_
tempo/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€
tempo/ExpandDims
ExpandDimsfeatures_tempotempo/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€T
tempo/ShapeShapetempo/ExpandDims:output:0*
T0*
_output_shapes
:c
tempo/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
tempo/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
tempo/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
tempo/strided_sliceStridedSlicetempo/Shape:output:0"tempo/strided_slice/stack:output:0$tempo/strided_slice/stack_1:output:0$tempo/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
tempo/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :З
tempo/Reshape/shapePacktempo/strided_slice:output:0tempo/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Г
tempo/ReshapeReshapetempo/ExpandDims:output:0tempo/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€a
valence/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Е
valence/ExpandDims
ExpandDimsfeatures_valencevalence/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
valence/ShapeShapevalence/ExpandDims:output:0*
T0*
_output_shapes
:e
valence/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
valence/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
valence/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
valence/strided_sliceStridedSlicevalence/Shape:output:0$valence/strided_slice/stack:output:0&valence/strided_slice/stack_1:output:0&valence/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
valence/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Н
valence/Reshape/shapePackvalence/strided_slice:output:0 valence/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Й
valence/ReshapeReshapevalence/ExpandDims:output:0valence/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Ф
concatConcatV2acousticness/Reshape:output:0danceability/Reshape:output:0duration/Reshape:output:0energy/Reshape:output:0!instrumentalness/Reshape:output:0key/Reshape:output:0liveness/Reshape:output:0loudness/Reshape:output:0speechiness/Reshape:output:0tempo/Reshape:output:0valence/Reshape:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*…
_input_shapesЈ
і:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:P L
#
_output_shapes
:€€€€€€€€€
%
_user_specified_namefeatures_id:ZV
#
_output_shapes
:€€€€€€€€€
/
_user_specified_namefeatures_acousticness:ZV
#
_output_shapes
:€€€€€€€€€
/
_user_specified_namefeatures_danceability:VR
#
_output_shapes
:€€€€€€€€€
+
_user_specified_namefeatures_duration:TP
#
_output_shapes
:€€€€€€€€€
)
_user_specified_namefeatures_energy:^Z
#
_output_shapes
:€€€€€€€€€
3
_user_specified_namefeatures_instrumentalness:QM
#
_output_shapes
:€€€€€€€€€
&
_user_specified_namefeatures_key:VR
#
_output_shapes
:€€€€€€€€€
+
_user_specified_namefeatures_liveness:VR
#
_output_shapes
:€€€€€€€€€
+
_user_specified_namefeatures_loudness:Y	U
#
_output_shapes
:€€€€€€€€€
.
_user_specified_namefeatures_speechiness:S
O
#
_output_shapes
:€€€€€€€€€
(
_user_specified_namefeatures_tempo:UQ
#
_output_shapes
:€€€€€€€€€
*
_user_specified_namefeatures_valence
ј
Х
(__inference_dense_1_layer_call_fn_989847

inputs
unknown:@@
	unknown_0:@
identityИҐStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_988225o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
»
Х
+__inference_sequential_layer_call_fn_989083
	inputs_id
inputs_acousticness
inputs_danceability
inputs_duration	
inputs_energy
inputs_instrumentalness

inputs_key	
inputs_liveness
inputs_loudness
inputs_speechiness
inputs_tempo
inputs_valence
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:	@А
	unknown_6:	А
	unknown_7:
АА
	unknown_8:	А
	unknown_9:
АА

unknown_10:	А

unknown_11:
АА

unknown_12:	А

unknown_13:	А

unknown_14:
identityИҐStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCall	inputs_idinputs_acousticnessinputs_danceabilityinputs_durationinputs_energyinputs_instrumentalness
inputs_keyinputs_livenessinputs_loudnessinputs_speechinessinputs_tempoinputs_valenceunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*'
Tin 
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*2
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_988341o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*й
_input_shapes„
‘:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
#
_output_shapes
:€€€€€€€€€
#
_user_specified_name	inputs_id:XT
#
_output_shapes
:€€€€€€€€€
-
_user_specified_nameinputs_acousticness:XT
#
_output_shapes
:€€€€€€€€€
-
_user_specified_nameinputs_danceability:TP
#
_output_shapes
:€€€€€€€€€
)
_user_specified_nameinputs_duration:RN
#
_output_shapes
:€€€€€€€€€
'
_user_specified_nameinputs_energy:\X
#
_output_shapes
:€€€€€€€€€
1
_user_specified_nameinputs_instrumentalness:OK
#
_output_shapes
:€€€€€€€€€
$
_user_specified_name
inputs_key:TP
#
_output_shapes
:€€€€€€€€€
)
_user_specified_nameinputs_liveness:TP
#
_output_shapes
:€€€€€€€€€
)
_user_specified_nameinputs_loudness:W	S
#
_output_shapes
:€€€€€€€€€
,
_user_specified_nameinputs_speechiness:Q
M
#
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs_tempo:SO
#
_output_shapes
:€€€€€€€€€
(
_user_specified_nameinputs_valence
ћ
Ѕ
+__inference_sequential_layer_call_fn_988376
id
acousticness
danceability
duration	

energy
instrumentalness
key	
liveness
loudness
speechiness	
tempo
valence
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:	@А
	unknown_6:	А
	unknown_7:
АА
	unknown_8:	А
	unknown_9:
АА

unknown_10:	А

unknown_11:
АА

unknown_12:	А

unknown_13:	А

unknown_14:
identityИҐStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallidacousticnessdanceabilitydurationenergyinstrumentalnesskeylivenessloudnessspeechinesstempovalenceunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*'
Tin 
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*2
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_988341o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*й
_input_shapes„
‘:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:G C
#
_output_shapes
:€€€€€€€€€

_user_specified_nameID:QM
#
_output_shapes
:€€€€€€€€€
&
_user_specified_nameacousticness:QM
#
_output_shapes
:€€€€€€€€€
&
_user_specified_namedanceability:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
duration:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameenergy:UQ
#
_output_shapes
:€€€€€€€€€
*
_user_specified_nameinstrumentalness:HD
#
_output_shapes
:€€€€€€€€€

_user_specified_namekey:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
liveness:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
loudness:P	L
#
_output_shapes
:€€€€€€€€€
%
_user_specified_namespeechiness:J
F
#
_output_shapes
:€€€€€€€€€

_user_specified_nametempo:LH
#
_output_shapes
:€€€€€€€€€
!
_user_specified_name	valence
ПГ
р
J__inference_dense_features_layer_call_and_return_conditional_losses_989689
features_id
features_acousticness
features_danceability
features_duration	
features_energy
features_instrumentalness
features_key	
features_liveness
features_loudness
features_speechiness
features_tempo
features_valence
identityf
acousticness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Ф
acousticness/ExpandDims
ExpandDimsfeatures_acousticness$acousticness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€b
acousticness/ShapeShape acousticness/ExpandDims:output:0*
T0*
_output_shapes
:j
 acousticness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"acousticness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"acousticness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
acousticness/strided_sliceStridedSliceacousticness/Shape:output:0)acousticness/strided_slice/stack:output:0+acousticness/strided_slice/stack_1:output:0+acousticness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
acousticness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ь
acousticness/Reshape/shapePack#acousticness/strided_slice:output:0%acousticness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ш
acousticness/ReshapeReshape acousticness/ExpandDims:output:0#acousticness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€f
danceability/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Ф
danceability/ExpandDims
ExpandDimsfeatures_danceability$danceability/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€b
danceability/ShapeShape danceability/ExpandDims:output:0*
T0*
_output_shapes
:j
 danceability/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: l
"danceability/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:l
"danceability/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Т
danceability/strided_sliceStridedSlicedanceability/Shape:output:0)danceability/strided_slice/stack:output:0+danceability/strided_slice/stack_1:output:0+danceability/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask^
danceability/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ь
danceability/Reshape/shapePack#danceability/strided_slice:output:0%danceability/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ш
danceability/ReshapeReshape danceability/ExpandDims:output:0#danceability/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€b
duration/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€И
duration/ExpandDims
ExpandDimsfeatures_duration duration/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:€€€€€€€€€t
duration/CastCastduration/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:€€€€€€€€€O
duration/ShapeShapeduration/Cast:y:0*
T0*
_output_shapes
:f
duration/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
duration/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
duration/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
duration/strided_sliceStridedSliceduration/Shape:output:0%duration/strided_slice/stack:output:0'duration/strided_slice/stack_1:output:0'duration/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
duration/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Р
duration/Reshape/shapePackduration/strided_slice:output:0!duration/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Б
duration/ReshapeReshapeduration/Cast:y:0duration/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
energy/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€В
energy/ExpandDims
ExpandDimsfeatures_energyenergy/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
energy/ShapeShapeenergy/ExpandDims:output:0*
T0*
_output_shapes
:d
energy/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
energy/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
energy/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ф
energy/strided_sliceStridedSliceenergy/Shape:output:0#energy/strided_slice/stack:output:0%energy/strided_slice/stack_1:output:0%energy/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
energy/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :К
energy/Reshape/shapePackenergy/strided_slice:output:0energy/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ж
energy/ReshapeReshapeenergy/ExpandDims:output:0energy/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€j
instrumentalness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€†
instrumentalness/ExpandDims
ExpandDimsfeatures_instrumentalness(instrumentalness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€j
instrumentalness/ShapeShape$instrumentalness/ExpandDims:output:0*
T0*
_output_shapes
:n
$instrumentalness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&instrumentalness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&instrumentalness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¶
instrumentalness/strided_sliceStridedSliceinstrumentalness/Shape:output:0-instrumentalness/strided_slice/stack:output:0/instrumentalness/strided_slice/stack_1:output:0/instrumentalness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskb
 instrumentalness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :®
instrumentalness/Reshape/shapePack'instrumentalness/strided_slice:output:0)instrumentalness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:§
instrumentalness/ReshapeReshape$instrumentalness/ExpandDims:output:0'instrumentalness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€]
key/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€y
key/ExpandDims
ExpandDimsfeatures_keykey/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:€€€€€€€€€j
key/CastCastkey/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:€€€€€€€€€E
	key/ShapeShapekey/Cast:y:0*
T0*
_output_shapes
:a
key/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
key/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
key/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:е
key/strided_sliceStridedSlicekey/Shape:output:0 key/strided_slice/stack:output:0"key/strided_slice/stack_1:output:0"key/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
key/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Б
key/Reshape/shapePackkey/strided_slice:output:0key/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:r
key/ReshapeReshapekey/Cast:y:0key/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€b
liveness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€И
liveness/ExpandDims
ExpandDimsfeatures_liveness liveness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€Z
liveness/ShapeShapeliveness/ExpandDims:output:0*
T0*
_output_shapes
:f
liveness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
liveness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
liveness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
liveness/strided_sliceStridedSliceliveness/Shape:output:0%liveness/strided_slice/stack:output:0'liveness/strided_slice/stack_1:output:0'liveness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
liveness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Р
liveness/Reshape/shapePackliveness/strided_slice:output:0!liveness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:М
liveness/ReshapeReshapeliveness/ExpandDims:output:0liveness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€b
loudness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€И
loudness/ExpandDims
ExpandDimsfeatures_loudness loudness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€Z
loudness/ShapeShapeloudness/ExpandDims:output:0*
T0*
_output_shapes
:f
loudness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
loudness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
loudness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ю
loudness/strided_sliceStridedSliceloudness/Shape:output:0%loudness/strided_slice/stack:output:0'loudness/strided_slice/stack_1:output:0'loudness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
loudness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Р
loudness/Reshape/shapePackloudness/strided_slice:output:0!loudness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:М
loudness/ReshapeReshapeloudness/ExpandDims:output:0loudness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€e
speechiness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€С
speechiness/ExpandDims
ExpandDimsfeatures_speechiness#speechiness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
speechiness/ShapeShapespeechiness/ExpandDims:output:0*
T0*
_output_shapes
:i
speechiness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: k
!speechiness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:k
!speechiness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Н
speechiness/strided_sliceStridedSlicespeechiness/Shape:output:0(speechiness/strided_slice/stack:output:0*speechiness/strided_slice/stack_1:output:0*speechiness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
speechiness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Щ
speechiness/Reshape/shapePack"speechiness/strided_slice:output:0$speechiness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Х
speechiness/ReshapeReshapespeechiness/ExpandDims:output:0"speechiness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€_
tempo/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€
tempo/ExpandDims
ExpandDimsfeatures_tempotempo/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€T
tempo/ShapeShapetempo/ExpandDims:output:0*
T0*
_output_shapes
:c
tempo/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: e
tempo/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:e
tempo/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:п
tempo/strided_sliceStridedSlicetempo/Shape:output:0"tempo/strided_slice/stack:output:0$tempo/strided_slice/stack_1:output:0$tempo/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
tempo/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :З
tempo/Reshape/shapePacktempo/strided_slice:output:0tempo/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Г
tempo/ReshapeReshapetempo/ExpandDims:output:0tempo/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€a
valence/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Е
valence/ExpandDims
ExpandDimsfeatures_valencevalence/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€X
valence/ShapeShapevalence/ExpandDims:output:0*
T0*
_output_shapes
:e
valence/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
valence/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
valence/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
valence/strided_sliceStridedSlicevalence/Shape:output:0$valence/strided_slice/stack:output:0&valence/strided_slice/stack_1:output:0&valence/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
valence/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Н
valence/Reshape/shapePackvalence/strided_slice:output:0 valence/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Й
valence/ReshapeReshapevalence/ExpandDims:output:0valence/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€V
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Ф
concatConcatV2acousticness/Reshape:output:0danceability/Reshape:output:0duration/Reshape:output:0energy/Reshape:output:0!instrumentalness/Reshape:output:0key/Reshape:output:0liveness/Reshape:output:0loudness/Reshape:output:0speechiness/Reshape:output:0tempo/Reshape:output:0valence/Reshape:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:€€€€€€€€€"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*…
_input_shapesЈ
і:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:P L
#
_output_shapes
:€€€€€€€€€
%
_user_specified_namefeatures_id:ZV
#
_output_shapes
:€€€€€€€€€
/
_user_specified_namefeatures_acousticness:ZV
#
_output_shapes
:€€€€€€€€€
/
_user_specified_namefeatures_danceability:VR
#
_output_shapes
:€€€€€€€€€
+
_user_specified_namefeatures_duration:TP
#
_output_shapes
:€€€€€€€€€
)
_user_specified_namefeatures_energy:^Z
#
_output_shapes
:€€€€€€€€€
3
_user_specified_namefeatures_instrumentalness:QM
#
_output_shapes
:€€€€€€€€€
&
_user_specified_namefeatures_key:VR
#
_output_shapes
:€€€€€€€€€
+
_user_specified_namefeatures_liveness:VR
#
_output_shapes
:€€€€€€€€€
+
_user_specified_namefeatures_loudness:Y	U
#
_output_shapes
:€€€€€€€€€
.
_user_specified_namefeatures_speechiness:S
O
#
_output_shapes
:€€€€€€€€€
(
_user_specified_namefeatures_tempo:UQ
#
_output_shapes
:€€€€€€€€€
*
_user_specified_namefeatures_valence
†
Ї
$__inference_signature_wrapper_989035
id
acousticness
danceability
duration	

energy
instrumentalness
key	
liveness
loudness
speechiness	
tempo
valence
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:	@А
	unknown_6:	А
	unknown_7:
АА
	unknown_8:	А
	unknown_9:
АА

unknown_10:	А

unknown_11:
АА

unknown_12:	А

unknown_13:	А

unknown_14:
identityИҐStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallidacousticnessdanceabilitydurationenergyinstrumentalnesskeylivenessloudnessspeechinesstempovalenceunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*'
Tin 
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*2
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__wrapped_model_988028o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*й
_input_shapes„
‘:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:G C
#
_output_shapes
:€€€€€€€€€

_user_specified_nameID:QM
#
_output_shapes
:€€€€€€€€€
&
_user_specified_nameacousticness:QM
#
_output_shapes
:€€€€€€€€€
&
_user_specified_namedanceability:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
duration:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameenergy:UQ
#
_output_shapes
:€€€€€€€€€
*
_user_specified_nameinstrumentalness:HD
#
_output_shapes
:€€€€€€€€€

_user_specified_namekey:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
liveness:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
loudness:P	L
#
_output_shapes
:€€€€€€€€€
%
_user_specified_namespeechiness:J
F
#
_output_shapes
:€€€€€€€€€

_user_specified_nametempo:LH
#
_output_shapes
:€€€€€€€€€
!
_user_specified_name	valence
£

х
C__inference_dense_7_layer_call_and_return_conditional_losses_990005

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Пl
д
__inference__traced_save_990201
file_prefix6
2savev2_sequential_dense_kernel_read_readvariableop4
0savev2_sequential_dense_bias_read_readvariableop8
4savev2_sequential_dense_1_kernel_read_readvariableop6
2savev2_sequential_dense_1_bias_read_readvariableop8
4savev2_sequential_dense_2_kernel_read_readvariableop6
2savev2_sequential_dense_2_bias_read_readvariableop8
4savev2_sequential_dense_3_kernel_read_readvariableop6
2savev2_sequential_dense_3_bias_read_readvariableop8
4savev2_sequential_dense_4_kernel_read_readvariableop6
2savev2_sequential_dense_4_bias_read_readvariableop8
4savev2_sequential_dense_5_kernel_read_readvariableop6
2savev2_sequential_dense_5_bias_read_readvariableop8
4savev2_sequential_dense_6_kernel_read_readvariableop6
2savev2_sequential_dense_6_bias_read_readvariableop8
4savev2_sequential_dense_7_kernel_read_readvariableop6
2savev2_sequential_dense_7_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableop=
9savev2_adam_m_sequential_dense_kernel_read_readvariableop=
9savev2_adam_v_sequential_dense_kernel_read_readvariableop;
7savev2_adam_m_sequential_dense_bias_read_readvariableop;
7savev2_adam_v_sequential_dense_bias_read_readvariableop?
;savev2_adam_m_sequential_dense_1_kernel_read_readvariableop?
;savev2_adam_v_sequential_dense_1_kernel_read_readvariableop=
9savev2_adam_m_sequential_dense_1_bias_read_readvariableop=
9savev2_adam_v_sequential_dense_1_bias_read_readvariableop?
;savev2_adam_m_sequential_dense_2_kernel_read_readvariableop?
;savev2_adam_v_sequential_dense_2_kernel_read_readvariableop=
9savev2_adam_m_sequential_dense_2_bias_read_readvariableop=
9savev2_adam_v_sequential_dense_2_bias_read_readvariableop?
;savev2_adam_m_sequential_dense_3_kernel_read_readvariableop?
;savev2_adam_v_sequential_dense_3_kernel_read_readvariableop=
9savev2_adam_m_sequential_dense_3_bias_read_readvariableop=
9savev2_adam_v_sequential_dense_3_bias_read_readvariableop?
;savev2_adam_m_sequential_dense_4_kernel_read_readvariableop?
;savev2_adam_v_sequential_dense_4_kernel_read_readvariableop=
9savev2_adam_m_sequential_dense_4_bias_read_readvariableop=
9savev2_adam_v_sequential_dense_4_bias_read_readvariableop?
;savev2_adam_m_sequential_dense_5_kernel_read_readvariableop?
;savev2_adam_v_sequential_dense_5_kernel_read_readvariableop=
9savev2_adam_m_sequential_dense_5_bias_read_readvariableop=
9savev2_adam_v_sequential_dense_5_bias_read_readvariableop?
;savev2_adam_m_sequential_dense_6_kernel_read_readvariableop?
;savev2_adam_v_sequential_dense_6_kernel_read_readvariableop=
9savev2_adam_m_sequential_dense_6_bias_read_readvariableop=
9savev2_adam_v_sequential_dense_6_bias_read_readvariableop?
;savev2_adam_m_sequential_dense_7_kernel_read_readvariableop?
;savev2_adam_v_sequential_dense_7_kernel_read_readvariableop=
9savev2_adam_m_sequential_dense_7_bias_read_readvariableop=
9savev2_adam_v_sequential_dense_7_bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1ИҐMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ∞
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*ў
valueѕBћ7B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH№
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:7*
dtype0*Б
valuexBv7B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ѓ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_sequential_dense_kernel_read_readvariableop0savev2_sequential_dense_bias_read_readvariableop4savev2_sequential_dense_1_kernel_read_readvariableop2savev2_sequential_dense_1_bias_read_readvariableop4savev2_sequential_dense_2_kernel_read_readvariableop2savev2_sequential_dense_2_bias_read_readvariableop4savev2_sequential_dense_3_kernel_read_readvariableop2savev2_sequential_dense_3_bias_read_readvariableop4savev2_sequential_dense_4_kernel_read_readvariableop2savev2_sequential_dense_4_bias_read_readvariableop4savev2_sequential_dense_5_kernel_read_readvariableop2savev2_sequential_dense_5_bias_read_readvariableop4savev2_sequential_dense_6_kernel_read_readvariableop2savev2_sequential_dense_6_bias_read_readvariableop4savev2_sequential_dense_7_kernel_read_readvariableop2savev2_sequential_dense_7_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableop9savev2_adam_m_sequential_dense_kernel_read_readvariableop9savev2_adam_v_sequential_dense_kernel_read_readvariableop7savev2_adam_m_sequential_dense_bias_read_readvariableop7savev2_adam_v_sequential_dense_bias_read_readvariableop;savev2_adam_m_sequential_dense_1_kernel_read_readvariableop;savev2_adam_v_sequential_dense_1_kernel_read_readvariableop9savev2_adam_m_sequential_dense_1_bias_read_readvariableop9savev2_adam_v_sequential_dense_1_bias_read_readvariableop;savev2_adam_m_sequential_dense_2_kernel_read_readvariableop;savev2_adam_v_sequential_dense_2_kernel_read_readvariableop9savev2_adam_m_sequential_dense_2_bias_read_readvariableop9savev2_adam_v_sequential_dense_2_bias_read_readvariableop;savev2_adam_m_sequential_dense_3_kernel_read_readvariableop;savev2_adam_v_sequential_dense_3_kernel_read_readvariableop9savev2_adam_m_sequential_dense_3_bias_read_readvariableop9savev2_adam_v_sequential_dense_3_bias_read_readvariableop;savev2_adam_m_sequential_dense_4_kernel_read_readvariableop;savev2_adam_v_sequential_dense_4_kernel_read_readvariableop9savev2_adam_m_sequential_dense_4_bias_read_readvariableop9savev2_adam_v_sequential_dense_4_bias_read_readvariableop;savev2_adam_m_sequential_dense_5_kernel_read_readvariableop;savev2_adam_v_sequential_dense_5_kernel_read_readvariableop9savev2_adam_m_sequential_dense_5_bias_read_readvariableop9savev2_adam_v_sequential_dense_5_bias_read_readvariableop;savev2_adam_m_sequential_dense_6_kernel_read_readvariableop;savev2_adam_v_sequential_dense_6_kernel_read_readvariableop9savev2_adam_m_sequential_dense_6_bias_read_readvariableop9savev2_adam_v_sequential_dense_6_bias_read_readvariableop;savev2_adam_m_sequential_dense_7_kernel_read_readvariableop;savev2_adam_v_sequential_dense_7_kernel_read_readvariableop9savev2_adam_m_sequential_dense_7_bias_read_readvariableop9savev2_adam_v_sequential_dense_7_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *E
dtypes;
927	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:≥
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*…
_input_shapesЈ
і: :@:@:@@:@:@@:@:	@А:А:
АА:А:
АА:А:
АА:А:	А:: : :@:@:@:@:@@:@@:@:@:@@:@@:@:@:	@А:	@А:А:А:
АА:
АА:А:А:
АА:
АА:А:А:
АА:
АА:А:А:	А:	А::: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@:%!

_output_shapes
:	@А:!

_output_shapes	
:А:&	"
 
_output_shapes
:
АА:!


_output_shapes	
:А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:%!

_output_shapes
:	А: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:@:$ 

_output_shapes

:@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@@:$ 

_output_shapes

:@@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@@:$ 

_output_shapes

:@@: 

_output_shapes
:@: 

_output_shapes
:@:%!

_output_shapes
:	@А:% !

_output_shapes
:	@А:!!

_output_shapes	
:А:!"

_output_shapes	
:А:&#"
 
_output_shapes
:
АА:&$"
 
_output_shapes
:
АА:!%

_output_shapes	
:А:!&

_output_shapes	
:А:&'"
 
_output_shapes
:
АА:&("
 
_output_shapes
:
АА:!)

_output_shapes	
:А:!*

_output_shapes	
:А:&+"
 
_output_shapes
:
АА:&,"
 
_output_shapes
:
АА:!-

_output_shapes	
:А:!.

_output_shapes	
:А:%/!

_output_shapes
:	А:%0!

_output_shapes
:	А: 1

_output_shapes
:: 2

_output_shapes
::3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :6

_output_shapes
: :7

_output_shapes
: 
Ґ

ц
C__inference_dense_3_layer_call_and_return_conditional_losses_988259

inputs1
matmul_readvariableop_resource:	@А.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
„B
ї
F__inference_sequential_layer_call_and_return_conditional_losses_988983
id
acousticness
danceability
duration	

energy
instrumentalness
key	
liveness
loudness
speechiness	
tempo
valence
dense_988941:@
dense_988943:@ 
dense_1_988946:@@
dense_1_988948:@ 
dense_2_988951:@@
dense_2_988953:@!
dense_3_988956:	@А
dense_3_988958:	А"
dense_4_988961:
АА
dense_4_988963:	А"
dense_5_988966:
АА
dense_5_988968:	А"
dense_6_988972:
АА
dense_6_988974:	А!
dense_7_988977:	А
dense_7_988979:
identityИҐdense/StatefulPartitionedCallҐdense_1/StatefulPartitionedCallҐdense_2/StatefulPartitionedCallҐdense_3/StatefulPartitionedCallҐdense_4/StatefulPartitionedCallҐdense_5/StatefulPartitionedCallҐdense_6/StatefulPartitionedCallҐdense_7/StatefulPartitionedCallҐdropout/StatefulPartitionedCallf
dense_features/CastCastacousticness*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€h
dense_features/Cast_1Castdanceability*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€b
dense_features/Cast_2Castenergy*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€l
dense_features/Cast_3Castinstrumentalness*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€d
dense_features/Cast_4Castliveness*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€d
dense_features/Cast_5Castloudness*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€g
dense_features/Cast_6Castspeechiness*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€a
dense_features/Cast_7Casttempo*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€c
dense_features/Cast_8Castvalence*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€Ћ
dense_features/PartitionedCallPartitionedCalliddense_features/Cast:y:0dense_features/Cast_1:y:0durationdense_features/Cast_2:y:0dense_features/Cast_3:y:0keydense_features/Cast_4:y:0dense_features/Cast_5:y:0dense_features/Cast_6:y:0dense_features/Cast_7:y:0dense_features/Cast_8:y:0*
Tin
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_dense_features_layer_call_and_return_conditional_losses_988638Е
dense/StatefulPartitionedCallStatefulPartitionedCall'dense_features/PartitionedCall:output:0dense_988941dense_988943*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_988208М
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_988946dense_1_988948*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_988225О
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_988951dense_2_988953*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_988242П
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_988956dense_3_988958*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_988259П
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_988961dense_4_988963*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_988276П
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_988966dense_5_988968*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_988293й
dropout/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_988416П
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_6_988972dense_6_988974*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_6_layer_call_and_return_conditional_losses_988317О
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_988977dense_7_988979*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_7_layer_call_and_return_conditional_losses_988334w
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ц
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*й
_input_shapes„
‘:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:G C
#
_output_shapes
:€€€€€€€€€

_user_specified_nameID:QM
#
_output_shapes
:€€€€€€€€€
&
_user_specified_nameacousticness:QM
#
_output_shapes
:€€€€€€€€€
&
_user_specified_namedanceability:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
duration:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameenergy:UQ
#
_output_shapes
:€€€€€€€€€
*
_user_specified_nameinstrumentalness:HD
#
_output_shapes
:€€€€€€€€€

_user_specified_namekey:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
liveness:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
loudness:P	L
#
_output_shapes
:€€€€€€€€€
%
_user_specified_namespeechiness:J
F
#
_output_shapes
:€€€€€€€€€

_user_specified_nametempo:LH
#
_output_shapes
:€€€€€€€€€
!
_user_specified_name	valence
Ш

т
A__inference_dense_layer_call_and_return_conditional_losses_988208

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ъ

ф
C__inference_dense_1_layer_call_and_return_conditional_losses_988225

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Ъ

ф
C__inference_dense_2_layer_call_and_return_conditional_losses_988242

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
¶

ч
C__inference_dense_6_layer_call_and_return_conditional_losses_988317

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
¶

ч
C__inference_dense_5_layer_call_and_return_conditional_losses_988293

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
фи
л
F__inference_sequential_layer_call_and_return_conditional_losses_989326
	inputs_id
inputs_acousticness
inputs_danceability
inputs_duration	
inputs_energy
inputs_instrumentalness

inputs_key	
inputs_liveness
inputs_loudness
inputs_speechiness
inputs_tempo
inputs_valence6
$dense_matmul_readvariableop_resource:@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@@5
'dense_1_biasadd_readvariableop_resource:@8
&dense_2_matmul_readvariableop_resource:@@5
'dense_2_biasadd_readvariableop_resource:@9
&dense_3_matmul_readvariableop_resource:	@А6
'dense_3_biasadd_readvariableop_resource:	А:
&dense_4_matmul_readvariableop_resource:
АА6
'dense_4_biasadd_readvariableop_resource:	А:
&dense_5_matmul_readvariableop_resource:
АА6
'dense_5_biasadd_readvariableop_resource:	А:
&dense_6_matmul_readvariableop_resource:
АА6
'dense_6_biasadd_readvariableop_resource:	А9
&dense_7_matmul_readvariableop_resource:	А5
'dense_7_biasadd_readvariableop_resource:
identityИҐdense/BiasAdd/ReadVariableOpҐdense/MatMul/ReadVariableOpҐdense_1/BiasAdd/ReadVariableOpҐdense_1/MatMul/ReadVariableOpҐdense_2/BiasAdd/ReadVariableOpҐdense_2/MatMul/ReadVariableOpҐdense_3/BiasAdd/ReadVariableOpҐdense_3/MatMul/ReadVariableOpҐdense_4/BiasAdd/ReadVariableOpҐdense_4/MatMul/ReadVariableOpҐdense_5/BiasAdd/ReadVariableOpҐdense_5/MatMul/ReadVariableOpҐdense_6/BiasAdd/ReadVariableOpҐdense_6/MatMul/ReadVariableOpҐdense_7/BiasAdd/ReadVariableOpҐdense_7/MatMul/ReadVariableOpm
dense_features/CastCastinputs_acousticness*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€o
dense_features/Cast_1Castinputs_danceability*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€i
dense_features/Cast_2Castinputs_energy*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€s
dense_features/Cast_3Castinputs_instrumentalness*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€k
dense_features/Cast_4Castinputs_liveness*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€k
dense_features/Cast_5Castinputs_loudness*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€n
dense_features/Cast_6Castinputs_speechiness*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€h
dense_features/Cast_7Castinputs_tempo*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€j
dense_features/Cast_8Castinputs_valence*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€u
*dense_features/acousticness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
&dense_features/acousticness/ExpandDims
ExpandDimsdense_features/Cast:y:03dense_features/acousticness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€А
!dense_features/acousticness/ShapeShape/dense_features/acousticness/ExpandDims:output:0*
T0*
_output_shapes
:y
/dense_features/acousticness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_features/acousticness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_features/acousticness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ё
)dense_features/acousticness/strided_sliceStridedSlice*dense_features/acousticness/Shape:output:08dense_features/acousticness/strided_slice/stack:output:0:dense_features/acousticness/strided_slice/stack_1:output:0:dense_features/acousticness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+dense_features/acousticness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :…
)dense_features/acousticness/Reshape/shapePack2dense_features/acousticness/strided_slice:output:04dense_features/acousticness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:≈
#dense_features/acousticness/ReshapeReshape/dense_features/acousticness/ExpandDims:output:02dense_features/acousticness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€u
*dense_features/danceability/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ґ
&dense_features/danceability/ExpandDims
ExpandDimsdense_features/Cast_1:y:03dense_features/danceability/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€А
!dense_features/danceability/ShapeShape/dense_features/danceability/ExpandDims:output:0*
T0*
_output_shapes
:y
/dense_features/danceability/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_features/danceability/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_features/danceability/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ё
)dense_features/danceability/strided_sliceStridedSlice*dense_features/danceability/Shape:output:08dense_features/danceability/strided_slice/stack:output:0:dense_features/danceability/strided_slice/stack_1:output:0:dense_features/danceability/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
+dense_features/danceability/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :…
)dense_features/danceability/Reshape/shapePack2dense_features/danceability/strided_slice:output:04dense_features/danceability/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:≈
#dense_features/danceability/ReshapeReshape/dense_features/danceability/ExpandDims:output:02dense_features/danceability/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€q
&dense_features/duration/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€§
"dense_features/duration/ExpandDims
ExpandDimsinputs_duration/dense_features/duration/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:€€€€€€€€€Т
dense_features/duration/CastCast+dense_features/duration/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:€€€€€€€€€m
dense_features/duration/ShapeShape dense_features/duration/Cast:y:0*
T0*
_output_shapes
:u
+dense_features/duration/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-dense_features/duration/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-dense_features/duration/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:…
%dense_features/duration/strided_sliceStridedSlice&dense_features/duration/Shape:output:04dense_features/duration/strided_slice/stack:output:06dense_features/duration/strided_slice/stack_1:output:06dense_features/duration/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'dense_features/duration/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :љ
%dense_features/duration/Reshape/shapePack.dense_features/duration/strided_slice:output:00dense_features/duration/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Ѓ
dense_features/duration/ReshapeReshape dense_features/duration/Cast:y:0.dense_features/duration/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€o
$dense_features/energy/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€™
 dense_features/energy/ExpandDims
ExpandDimsdense_features/Cast_2:y:0-dense_features/energy/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€t
dense_features/energy/ShapeShape)dense_features/energy/ExpandDims:output:0*
T0*
_output_shapes
:s
)dense_features/energy/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+dense_features/energy/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+dense_features/energy/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:њ
#dense_features/energy/strided_sliceStridedSlice$dense_features/energy/Shape:output:02dense_features/energy/strided_slice/stack:output:04dense_features/energy/strided_slice/stack_1:output:04dense_features/energy/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%dense_features/energy/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ј
#dense_features/energy/Reshape/shapePack,dense_features/energy/strided_slice:output:0.dense_features/energy/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:≥
dense_features/energy/ReshapeReshape)dense_features/energy/ExpandDims:output:0,dense_features/energy/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€y
.dense_features/instrumentalness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Њ
*dense_features/instrumentalness/ExpandDims
ExpandDimsdense_features/Cast_3:y:07dense_features/instrumentalness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€И
%dense_features/instrumentalness/ShapeShape3dense_features/instrumentalness/ExpandDims:output:0*
T0*
_output_shapes
:}
3dense_features/instrumentalness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5dense_features/instrumentalness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5dense_features/instrumentalness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:с
-dense_features/instrumentalness/strided_sliceStridedSlice.dense_features/instrumentalness/Shape:output:0<dense_features/instrumentalness/strided_slice/stack:output:0>dense_features/instrumentalness/strided_slice/stack_1:output:0>dense_features/instrumentalness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
/dense_features/instrumentalness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :’
-dense_features/instrumentalness/Reshape/shapePack6dense_features/instrumentalness/strided_slice:output:08dense_features/instrumentalness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:—
'dense_features/instrumentalness/ReshapeReshape3dense_features/instrumentalness/ExpandDims:output:06dense_features/instrumentalness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€l
!dense_features/key/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Х
dense_features/key/ExpandDims
ExpandDims
inputs_key*dense_features/key/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:€€€€€€€€€И
dense_features/key/CastCast&dense_features/key/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:€€€€€€€€€c
dense_features/key/ShapeShapedense_features/key/Cast:y:0*
T0*
_output_shapes
:p
&dense_features/key/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(dense_features/key/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(dense_features/key/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:∞
 dense_features/key/strided_sliceStridedSlice!dense_features/key/Shape:output:0/dense_features/key/strided_slice/stack:output:01dense_features/key/strided_slice/stack_1:output:01dense_features/key/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"dense_features/key/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ѓ
 dense_features/key/Reshape/shapePack)dense_features/key/strided_slice:output:0+dense_features/key/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Я
dense_features/key/ReshapeReshapedense_features/key/Cast:y:0)dense_features/key/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€q
&dense_features/liveness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Ѓ
"dense_features/liveness/ExpandDims
ExpandDimsdense_features/Cast_4:y:0/dense_features/liveness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€x
dense_features/liveness/ShapeShape+dense_features/liveness/ExpandDims:output:0*
T0*
_output_shapes
:u
+dense_features/liveness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-dense_features/liveness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-dense_features/liveness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:…
%dense_features/liveness/strided_sliceStridedSlice&dense_features/liveness/Shape:output:04dense_features/liveness/strided_slice/stack:output:06dense_features/liveness/strided_slice/stack_1:output:06dense_features/liveness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'dense_features/liveness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :љ
%dense_features/liveness/Reshape/shapePack.dense_features/liveness/strided_slice:output:00dense_features/liveness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:є
dense_features/liveness/ReshapeReshape+dense_features/liveness/ExpandDims:output:0.dense_features/liveness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€q
&dense_features/loudness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Ѓ
"dense_features/loudness/ExpandDims
ExpandDimsdense_features/Cast_5:y:0/dense_features/loudness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€x
dense_features/loudness/ShapeShape+dense_features/loudness/ExpandDims:output:0*
T0*
_output_shapes
:u
+dense_features/loudness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-dense_features/loudness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-dense_features/loudness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:…
%dense_features/loudness/strided_sliceStridedSlice&dense_features/loudness/Shape:output:04dense_features/loudness/strided_slice/stack:output:06dense_features/loudness/strided_slice/stack_1:output:06dense_features/loudness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maski
'dense_features/loudness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :љ
%dense_features/loudness/Reshape/shapePack.dense_features/loudness/strided_slice:output:00dense_features/loudness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:є
dense_features/loudness/ReshapeReshape+dense_features/loudness/ExpandDims:output:0.dense_features/loudness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€t
)dense_features/speechiness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€і
%dense_features/speechiness/ExpandDims
ExpandDimsdense_features/Cast_6:y:02dense_features/speechiness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€~
 dense_features/speechiness/ShapeShape.dense_features/speechiness/ExpandDims:output:0*
T0*
_output_shapes
:x
.dense_features/speechiness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0dense_features/speechiness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0dense_features/speechiness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ў
(dense_features/speechiness/strided_sliceStridedSlice)dense_features/speechiness/Shape:output:07dense_features/speechiness/strided_slice/stack:output:09dense_features/speechiness/strided_slice/stack_1:output:09dense_features/speechiness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*dense_features/speechiness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :∆
(dense_features/speechiness/Reshape/shapePack1dense_features/speechiness/strided_slice:output:03dense_features/speechiness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:¬
"dense_features/speechiness/ReshapeReshape.dense_features/speechiness/ExpandDims:output:01dense_features/speechiness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€n
#dense_features/tempo/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€®
dense_features/tempo/ExpandDims
ExpandDimsdense_features/Cast_7:y:0,dense_features/tempo/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€r
dense_features/tempo/ShapeShape(dense_features/tempo/ExpandDims:output:0*
T0*
_output_shapes
:r
(dense_features/tempo/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*dense_features/tempo/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*dense_features/tempo/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ї
"dense_features/tempo/strided_sliceStridedSlice#dense_features/tempo/Shape:output:01dense_features/tempo/strided_slice/stack:output:03dense_features/tempo/strided_slice/stack_1:output:03dense_features/tempo/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$dense_features/tempo/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :і
"dense_features/tempo/Reshape/shapePack+dense_features/tempo/strided_slice:output:0-dense_features/tempo/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:∞
dense_features/tempo/ReshapeReshape(dense_features/tempo/ExpandDims:output:0+dense_features/tempo/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€p
%dense_features/valence/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ђ
!dense_features/valence/ExpandDims
ExpandDimsdense_features/Cast_8:y:0.dense_features/valence/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€v
dense_features/valence/ShapeShape*dense_features/valence/ExpandDims:output:0*
T0*
_output_shapes
:t
*dense_features/valence/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
,dense_features/valence/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:v
,dense_features/valence/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ƒ
$dense_features/valence/strided_sliceStridedSlice%dense_features/valence/Shape:output:03dense_features/valence/strided_slice/stack:output:05dense_features/valence/strided_slice/stack_1:output:05dense_features/valence/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskh
&dense_features/valence/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ї
$dense_features/valence/Reshape/shapePack-dense_features/valence/strided_slice:output:0/dense_features/valence/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:ґ
dense_features/valence/ReshapeReshape*dense_features/valence/ExpandDims:output:0-dense_features/valence/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€e
dense_features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€„
dense_features/concatConcatV2,dense_features/acousticness/Reshape:output:0,dense_features/danceability/Reshape:output:0(dense_features/duration/Reshape:output:0&dense_features/energy/Reshape:output:00dense_features/instrumentalness/Reshape:output:0#dense_features/key/Reshape:output:0(dense_features/liveness/Reshape:output:0(dense_features/loudness/Reshape:output:0+dense_features/speechiness/Reshape:output:0%dense_features/tempo/Reshape:output:0'dense_features/valence/Reshape:output:0#dense_features/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€А
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Н
dense/MatMulMatMuldense_features/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0И
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@\

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Д
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0Л
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@В
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0О
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@`
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Д
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0Н
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@В
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0О
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@`
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Е
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype0О
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АГ
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0П
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аa
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€АЖ
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0О
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АГ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0П
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аa
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€АЖ
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0О
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АГ
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0П
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аa
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аk
dropout/IdentityIdentitydense_5/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€АЖ
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Н
dense_6/MatMulMatMuldropout/Identity:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АГ
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0П
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аa
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€АЕ
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Н
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€В
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0О
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€f
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€h
IdentityIdentitydense_7/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ 
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*й
_input_shapes„
‘:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:N J
#
_output_shapes
:€€€€€€€€€
#
_user_specified_name	inputs_id:XT
#
_output_shapes
:€€€€€€€€€
-
_user_specified_nameinputs_acousticness:XT
#
_output_shapes
:€€€€€€€€€
-
_user_specified_nameinputs_danceability:TP
#
_output_shapes
:€€€€€€€€€
)
_user_specified_nameinputs_duration:RN
#
_output_shapes
:€€€€€€€€€
'
_user_specified_nameinputs_energy:\X
#
_output_shapes
:€€€€€€€€€
1
_user_specified_nameinputs_instrumentalness:OK
#
_output_shapes
:€€€€€€€€€
$
_user_specified_name
inputs_key:TP
#
_output_shapes
:€€€€€€€€€
)
_user_specified_nameinputs_liveness:TP
#
_output_shapes
:€€€€€€€€€
)
_user_specified_nameinputs_loudness:W	S
#
_output_shapes
:€€€€€€€€€
,
_user_specified_nameinputs_speechiness:Q
M
#
_output_shapes
:€€€€€€€€€
&
_user_specified_nameinputs_tempo:SO
#
_output_shapes
:€€€€€€€€€
(
_user_specified_nameinputs_valence
«
Ш
(__inference_dense_4_layer_call_fn_989907

inputs
unknown:
АА
	unknown_0:	А
identityИҐStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_988276p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
¶

ч
C__inference_dense_4_layer_call_and_return_conditional_losses_989918

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:€€€€€€€€€Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
∞М
“
!__inference__wrapped_model_988028
id
acousticness
danceability
duration	

energy
instrumentalness
key	
liveness
loudness
speechiness	
tempo
valenceA
/sequential_dense_matmul_readvariableop_resource:@>
0sequential_dense_biasadd_readvariableop_resource:@C
1sequential_dense_1_matmul_readvariableop_resource:@@@
2sequential_dense_1_biasadd_readvariableop_resource:@C
1sequential_dense_2_matmul_readvariableop_resource:@@@
2sequential_dense_2_biasadd_readvariableop_resource:@D
1sequential_dense_3_matmul_readvariableop_resource:	@АA
2sequential_dense_3_biasadd_readvariableop_resource:	АE
1sequential_dense_4_matmul_readvariableop_resource:
ААA
2sequential_dense_4_biasadd_readvariableop_resource:	АE
1sequential_dense_5_matmul_readvariableop_resource:
ААA
2sequential_dense_5_biasadd_readvariableop_resource:	АE
1sequential_dense_6_matmul_readvariableop_resource:
ААA
2sequential_dense_6_biasadd_readvariableop_resource:	АD
1sequential_dense_7_matmul_readvariableop_resource:	А@
2sequential_dense_7_biasadd_readvariableop_resource:
identityИҐ'sequential/dense/BiasAdd/ReadVariableOpҐ&sequential/dense/MatMul/ReadVariableOpҐ)sequential/dense_1/BiasAdd/ReadVariableOpҐ(sequential/dense_1/MatMul/ReadVariableOpҐ)sequential/dense_2/BiasAdd/ReadVariableOpҐ(sequential/dense_2/MatMul/ReadVariableOpҐ)sequential/dense_3/BiasAdd/ReadVariableOpҐ(sequential/dense_3/MatMul/ReadVariableOpҐ)sequential/dense_4/BiasAdd/ReadVariableOpҐ(sequential/dense_4/MatMul/ReadVariableOpҐ)sequential/dense_5/BiasAdd/ReadVariableOpҐ(sequential/dense_5/MatMul/ReadVariableOpҐ)sequential/dense_6/BiasAdd/ReadVariableOpҐ(sequential/dense_6/MatMul/ReadVariableOpҐ)sequential/dense_7/BiasAdd/ReadVariableOpҐ(sequential/dense_7/MatMul/ReadVariableOpq
sequential/dense_features/CastCastacousticness*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€s
 sequential/dense_features/Cast_1Castdanceability*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€m
 sequential/dense_features/Cast_2Castenergy*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€w
 sequential/dense_features/Cast_3Castinstrumentalness*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€o
 sequential/dense_features/Cast_4Castliveness*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€o
 sequential/dense_features/Cast_5Castloudness*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€r
 sequential/dense_features/Cast_6Castspeechiness*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€l
 sequential/dense_features/Cast_7Casttempo*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€n
 sequential/dense_features/Cast_8Castvalence*

DstT0*

SrcT0*#
_output_shapes
:€€€€€€€€€А
5sequential/dense_features/acousticness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€’
1sequential/dense_features/acousticness/ExpandDims
ExpandDims"sequential/dense_features/Cast:y:0>sequential/dense_features/acousticness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ц
,sequential/dense_features/acousticness/ShapeShape:sequential/dense_features/acousticness/ExpandDims:output:0*
T0*
_output_shapes
:Д
:sequential/dense_features/acousticness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ж
<sequential/dense_features/acousticness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ж
<sequential/dense_features/acousticness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ф
4sequential/dense_features/acousticness/strided_sliceStridedSlice5sequential/dense_features/acousticness/Shape:output:0Csequential/dense_features/acousticness/strided_slice/stack:output:0Esequential/dense_features/acousticness/strided_slice/stack_1:output:0Esequential/dense_features/acousticness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
6sequential/dense_features/acousticness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :к
4sequential/dense_features/acousticness/Reshape/shapePack=sequential/dense_features/acousticness/strided_slice:output:0?sequential/dense_features/acousticness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:ж
.sequential/dense_features/acousticness/ReshapeReshape:sequential/dense_features/acousticness/ExpandDims:output:0=sequential/dense_features/acousticness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€А
5sequential/dense_features/danceability/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€„
1sequential/dense_features/danceability/ExpandDims
ExpandDims$sequential/dense_features/Cast_1:y:0>sequential/dense_features/danceability/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ц
,sequential/dense_features/danceability/ShapeShape:sequential/dense_features/danceability/ExpandDims:output:0*
T0*
_output_shapes
:Д
:sequential/dense_features/danceability/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Ж
<sequential/dense_features/danceability/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Ж
<sequential/dense_features/danceability/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ф
4sequential/dense_features/danceability/strided_sliceStridedSlice5sequential/dense_features/danceability/Shape:output:0Csequential/dense_features/danceability/strided_slice/stack:output:0Esequential/dense_features/danceability/strided_slice/stack_1:output:0Esequential/dense_features/danceability/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskx
6sequential/dense_features/danceability/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :к
4sequential/dense_features/danceability/Reshape/shapePack=sequential/dense_features/danceability/strided_slice:output:0?sequential/dense_features/danceability/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:ж
.sequential/dense_features/danceability/ReshapeReshape:sequential/dense_features/danceability/ExpandDims:output:0=sequential/dense_features/danceability/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€|
1sequential/dense_features/duration/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€≥
-sequential/dense_features/duration/ExpandDims
ExpandDimsduration:sequential/dense_features/duration/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:€€€€€€€€€®
'sequential/dense_features/duration/CastCast6sequential/dense_features/duration/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:€€€€€€€€€Г
(sequential/dense_features/duration/ShapeShape+sequential/dense_features/duration/Cast:y:0*
T0*
_output_shapes
:А
6sequential/dense_features/duration/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: В
8sequential/dense_features/duration/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:В
8sequential/dense_features/duration/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:А
0sequential/dense_features/duration/strided_sliceStridedSlice1sequential/dense_features/duration/Shape:output:0?sequential/dense_features/duration/strided_slice/stack:output:0Asequential/dense_features/duration/strided_slice/stack_1:output:0Asequential/dense_features/duration/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
2sequential/dense_features/duration/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ё
0sequential/dense_features/duration/Reshape/shapePack9sequential/dense_features/duration/strided_slice:output:0;sequential/dense_features/duration/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:ѕ
*sequential/dense_features/duration/ReshapeReshape+sequential/dense_features/duration/Cast:y:09sequential/dense_features/duration/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€z
/sequential/dense_features/energy/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Ћ
+sequential/dense_features/energy/ExpandDims
ExpandDims$sequential/dense_features/Cast_2:y:08sequential/dense_features/energy/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€К
&sequential/dense_features/energy/ShapeShape4sequential/dense_features/energy/ExpandDims:output:0*
T0*
_output_shapes
:~
4sequential/dense_features/energy/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: А
6sequential/dense_features/energy/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:А
6sequential/dense_features/energy/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ц
.sequential/dense_features/energy/strided_sliceStridedSlice/sequential/dense_features/energy/Shape:output:0=sequential/dense_features/energy/strided_slice/stack:output:0?sequential/dense_features/energy/strided_slice/stack_1:output:0?sequential/dense_features/energy/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskr
0sequential/dense_features/energy/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Ў
.sequential/dense_features/energy/Reshape/shapePack7sequential/dense_features/energy/strided_slice:output:09sequential/dense_features/energy/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:‘
(sequential/dense_features/energy/ReshapeReshape4sequential/dense_features/energy/ExpandDims:output:07sequential/dense_features/energy/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€Д
9sequential/dense_features/instrumentalness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€я
5sequential/dense_features/instrumentalness/ExpandDims
ExpandDims$sequential/dense_features/Cast_3:y:0Bsequential/dense_features/instrumentalness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ю
0sequential/dense_features/instrumentalness/ShapeShape>sequential/dense_features/instrumentalness/ExpandDims:output:0*
T0*
_output_shapes
:И
>sequential/dense_features/instrumentalness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: К
@sequential/dense_features/instrumentalness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:К
@sequential/dense_features/instrumentalness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:®
8sequential/dense_features/instrumentalness/strided_sliceStridedSlice9sequential/dense_features/instrumentalness/Shape:output:0Gsequential/dense_features/instrumentalness/strided_slice/stack:output:0Isequential/dense_features/instrumentalness/strided_slice/stack_1:output:0Isequential/dense_features/instrumentalness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
:sequential/dense_features/instrumentalness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ц
8sequential/dense_features/instrumentalness/Reshape/shapePackAsequential/dense_features/instrumentalness/strided_slice:output:0Csequential/dense_features/instrumentalness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:т
2sequential/dense_features/instrumentalness/ReshapeReshape>sequential/dense_features/instrumentalness/ExpandDims:output:0Asequential/dense_features/instrumentalness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€w
,sequential/dense_features/key/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€§
(sequential/dense_features/key/ExpandDims
ExpandDimskey5sequential/dense_features/key/ExpandDims/dim:output:0*
T0	*'
_output_shapes
:€€€€€€€€€Ю
"sequential/dense_features/key/CastCast1sequential/dense_features/key/ExpandDims:output:0*

DstT0*

SrcT0	*'
_output_shapes
:€€€€€€€€€y
#sequential/dense_features/key/ShapeShape&sequential/dense_features/key/Cast:y:0*
T0*
_output_shapes
:{
1sequential/dense_features/key/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3sequential/dense_features/key/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3sequential/dense_features/key/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:з
+sequential/dense_features/key/strided_sliceStridedSlice,sequential/dense_features/key/Shape:output:0:sequential/dense_features/key/strided_slice/stack:output:0<sequential/dense_features/key/strided_slice/stack_1:output:0<sequential/dense_features/key/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masko
-sequential/dense_features/key/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ѕ
+sequential/dense_features/key/Reshape/shapePack4sequential/dense_features/key/strided_slice:output:06sequential/dense_features/key/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:ј
%sequential/dense_features/key/ReshapeReshape&sequential/dense_features/key/Cast:y:04sequential/dense_features/key/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€|
1sequential/dense_features/liveness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ѕ
-sequential/dense_features/liveness/ExpandDims
ExpandDims$sequential/dense_features/Cast_4:y:0:sequential/dense_features/liveness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€О
(sequential/dense_features/liveness/ShapeShape6sequential/dense_features/liveness/ExpandDims:output:0*
T0*
_output_shapes
:А
6sequential/dense_features/liveness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: В
8sequential/dense_features/liveness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:В
8sequential/dense_features/liveness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:А
0sequential/dense_features/liveness/strided_sliceStridedSlice1sequential/dense_features/liveness/Shape:output:0?sequential/dense_features/liveness/strided_slice/stack:output:0Asequential/dense_features/liveness/strided_slice/stack_1:output:0Asequential/dense_features/liveness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
2sequential/dense_features/liveness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ё
0sequential/dense_features/liveness/Reshape/shapePack9sequential/dense_features/liveness/strided_slice:output:0;sequential/dense_features/liveness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Џ
*sequential/dense_features/liveness/ReshapeReshape6sequential/dense_features/liveness/ExpandDims:output:09sequential/dense_features/liveness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€|
1sequential/dense_features/loudness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ѕ
-sequential/dense_features/loudness/ExpandDims
ExpandDims$sequential/dense_features/Cast_5:y:0:sequential/dense_features/loudness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€О
(sequential/dense_features/loudness/ShapeShape6sequential/dense_features/loudness/ExpandDims:output:0*
T0*
_output_shapes
:А
6sequential/dense_features/loudness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: В
8sequential/dense_features/loudness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:В
8sequential/dense_features/loudness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:А
0sequential/dense_features/loudness/strided_sliceStridedSlice1sequential/dense_features/loudness/Shape:output:0?sequential/dense_features/loudness/strided_slice/stack:output:0Asequential/dense_features/loudness/strided_slice/stack_1:output:0Asequential/dense_features/loudness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskt
2sequential/dense_features/loudness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :ё
0sequential/dense_features/loudness/Reshape/shapePack9sequential/dense_features/loudness/strided_slice:output:0;sequential/dense_features/loudness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:Џ
*sequential/dense_features/loudness/ReshapeReshape6sequential/dense_features/loudness/ExpandDims:output:09sequential/dense_features/loudness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€
4sequential/dense_features/speechiness/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€’
0sequential/dense_features/speechiness/ExpandDims
ExpandDims$sequential/dense_features/Cast_6:y:0=sequential/dense_features/speechiness/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€Ф
+sequential/dense_features/speechiness/ShapeShape9sequential/dense_features/speechiness/ExpandDims:output:0*
T0*
_output_shapes
:Г
9sequential/dense_features/speechiness/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Е
;sequential/dense_features/speechiness/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Е
;sequential/dense_features/speechiness/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:П
3sequential/dense_features/speechiness/strided_sliceStridedSlice4sequential/dense_features/speechiness/Shape:output:0Bsequential/dense_features/speechiness/strided_slice/stack:output:0Dsequential/dense_features/speechiness/strided_slice/stack_1:output:0Dsequential/dense_features/speechiness/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
5sequential/dense_features/speechiness/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :з
3sequential/dense_features/speechiness/Reshape/shapePack<sequential/dense_features/speechiness/strided_slice:output:0>sequential/dense_features/speechiness/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:г
-sequential/dense_features/speechiness/ReshapeReshape9sequential/dense_features/speechiness/ExpandDims:output:0<sequential/dense_features/speechiness/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€y
.sequential/dense_features/tempo/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€…
*sequential/dense_features/tempo/ExpandDims
ExpandDims$sequential/dense_features/Cast_7:y:07sequential/dense_features/tempo/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€И
%sequential/dense_features/tempo/ShapeShape3sequential/dense_features/tempo/ExpandDims:output:0*
T0*
_output_shapes
:}
3sequential/dense_features/tempo/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5sequential/dense_features/tempo/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5sequential/dense_features/tempo/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:с
-sequential/dense_features/tempo/strided_sliceStridedSlice.sequential/dense_features/tempo/Shape:output:0<sequential/dense_features/tempo/strided_slice/stack:output:0>sequential/dense_features/tempo/strided_slice/stack_1:output:0>sequential/dense_features/tempo/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskq
/sequential/dense_features/tempo/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :’
-sequential/dense_features/tempo/Reshape/shapePack6sequential/dense_features/tempo/strided_slice:output:08sequential/dense_features/tempo/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:—
'sequential/dense_features/tempo/ReshapeReshape3sequential/dense_features/tempo/ExpandDims:output:06sequential/dense_features/tempo/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€{
0sequential/dense_features/valence/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€Ќ
,sequential/dense_features/valence/ExpandDims
ExpandDims$sequential/dense_features/Cast_8:y:09sequential/dense_features/valence/ExpandDims/dim:output:0*
T0*'
_output_shapes
:€€€€€€€€€М
'sequential/dense_features/valence/ShapeShape5sequential/dense_features/valence/ExpandDims:output:0*
T0*
_output_shapes
:
5sequential/dense_features/valence/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: Б
7sequential/dense_features/valence/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:Б
7sequential/dense_features/valence/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ы
/sequential/dense_features/valence/strided_sliceStridedSlice0sequential/dense_features/valence/Shape:output:0>sequential/dense_features/valence/strided_slice/stack:output:0@sequential/dense_features/valence/strided_slice/stack_1:output:0@sequential/dense_features/valence/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masks
1sequential/dense_features/valence/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :џ
/sequential/dense_features/valence/Reshape/shapePack8sequential/dense_features/valence/strided_slice:output:0:sequential/dense_features/valence/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:„
)sequential/dense_features/valence/ReshapeReshape5sequential/dense_features/valence/ExpandDims:output:08sequential/dense_features/valence/Reshape/shape:output:0*
T0*'
_output_shapes
:€€€€€€€€€p
%sequential/dense_features/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€ж
 sequential/dense_features/concatConcatV27sequential/dense_features/acousticness/Reshape:output:07sequential/dense_features/danceability/Reshape:output:03sequential/dense_features/duration/Reshape:output:01sequential/dense_features/energy/Reshape:output:0;sequential/dense_features/instrumentalness/Reshape:output:0.sequential/dense_features/key/Reshape:output:03sequential/dense_features/liveness/Reshape:output:03sequential/dense_features/loudness/Reshape:output:06sequential/dense_features/speechiness/Reshape:output:00sequential/dense_features/tempo/Reshape:output:02sequential/dense_features/valence/Reshape:output:0.sequential/dense_features/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€Ц
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ѓ
sequential/dense/MatMulMatMul)sequential/dense_features/concat:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Ф
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0©
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@r
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ъ
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0ђ
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Ш
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ѓ
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@v
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ъ
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0Ѓ
sequential/dense_2/MatMulMatMul%sequential/dense_1/Relu:activations:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@Ш
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ѓ
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@v
sequential/dense_2/ReluRelu#sequential/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@Ы
(sequential/dense_3/MatMul/ReadVariableOpReadVariableOp1sequential_dense_3_matmul_readvariableop_resource*
_output_shapes
:	@А*
dtype0ѓ
sequential/dense_3/MatMulMatMul%sequential/dense_2/Relu:activations:00sequential/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АЩ
)sequential/dense_3/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0∞
sequential/dense_3/BiasAddBiasAdd#sequential/dense_3/MatMul:product:01sequential/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аw
sequential/dense_3/ReluRelu#sequential/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€АЬ
(sequential/dense_4/MatMul/ReadVariableOpReadVariableOp1sequential_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0ѓ
sequential/dense_4/MatMulMatMul%sequential/dense_3/Relu:activations:00sequential/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АЩ
)sequential/dense_4/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0∞
sequential/dense_4/BiasAddBiasAdd#sequential/dense_4/MatMul:product:01sequential/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аw
sequential/dense_4/ReluRelu#sequential/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€АЬ
(sequential/dense_5/MatMul/ReadVariableOpReadVariableOp1sequential_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0ѓ
sequential/dense_5/MatMulMatMul%sequential/dense_4/Relu:activations:00sequential/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АЩ
)sequential/dense_5/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0∞
sequential/dense_5/BiasAddBiasAdd#sequential/dense_5/MatMul:product:01sequential/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аw
sequential/dense_5/ReluRelu#sequential/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€АБ
sequential/dropout/IdentityIdentity%sequential/dense_5/Relu:activations:0*
T0*(
_output_shapes
:€€€€€€€€€АЬ
(sequential/dense_6/MatMul/ReadVariableOpReadVariableOp1sequential_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Ѓ
sequential/dense_6/MatMulMatMul$sequential/dropout/Identity:output:00sequential/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€АЩ
)sequential/dense_6/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0∞
sequential/dense_6/BiasAddBiasAdd#sequential/dense_6/MatMul:product:01sequential/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:€€€€€€€€€Аw
sequential/dense_6/ReluRelu#sequential/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:€€€€€€€€€АЫ
(sequential/dense_7/MatMul/ReadVariableOpReadVariableOp1sequential_dense_7_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ѓ
sequential/dense_7/MatMulMatMul%sequential/dense_6/Relu:activations:00sequential/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€Ш
)sequential/dense_7/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ѓ
sequential/dense_7/BiasAddBiasAdd#sequential/dense_7/MatMul:product:01sequential/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€|
sequential/dense_7/SoftmaxSoftmax#sequential/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€s
IdentityIdentity$sequential/dense_7/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:€€€€€€€€€ъ
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp*^sequential/dense_3/BiasAdd/ReadVariableOp)^sequential/dense_3/MatMul/ReadVariableOp*^sequential/dense_4/BiasAdd/ReadVariableOp)^sequential/dense_4/MatMul/ReadVariableOp*^sequential/dense_5/BiasAdd/ReadVariableOp)^sequential/dense_5/MatMul/ReadVariableOp*^sequential/dense_6/BiasAdd/ReadVariableOp)^sequential/dense_6/MatMul/ReadVariableOp*^sequential/dense_7/BiasAdd/ReadVariableOp)^sequential/dense_7/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*й
_input_shapes„
‘:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€: : : : : : : : : : : : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp2V
)sequential/dense_3/BiasAdd/ReadVariableOp)sequential/dense_3/BiasAdd/ReadVariableOp2T
(sequential/dense_3/MatMul/ReadVariableOp(sequential/dense_3/MatMul/ReadVariableOp2V
)sequential/dense_4/BiasAdd/ReadVariableOp)sequential/dense_4/BiasAdd/ReadVariableOp2T
(sequential/dense_4/MatMul/ReadVariableOp(sequential/dense_4/MatMul/ReadVariableOp2V
)sequential/dense_5/BiasAdd/ReadVariableOp)sequential/dense_5/BiasAdd/ReadVariableOp2T
(sequential/dense_5/MatMul/ReadVariableOp(sequential/dense_5/MatMul/ReadVariableOp2V
)sequential/dense_6/BiasAdd/ReadVariableOp)sequential/dense_6/BiasAdd/ReadVariableOp2T
(sequential/dense_6/MatMul/ReadVariableOp(sequential/dense_6/MatMul/ReadVariableOp2V
)sequential/dense_7/BiasAdd/ReadVariableOp)sequential/dense_7/BiasAdd/ReadVariableOp2T
(sequential/dense_7/MatMul/ReadVariableOp(sequential/dense_7/MatMul/ReadVariableOp:G C
#
_output_shapes
:€€€€€€€€€

_user_specified_nameID:QM
#
_output_shapes
:€€€€€€€€€
&
_user_specified_nameacousticness:QM
#
_output_shapes
:€€€€€€€€€
&
_user_specified_namedanceability:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
duration:KG
#
_output_shapes
:€€€€€€€€€
 
_user_specified_nameenergy:UQ
#
_output_shapes
:€€€€€€€€€
*
_user_specified_nameinstrumentalness:HD
#
_output_shapes
:€€€€€€€€€

_user_specified_namekey:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
liveness:MI
#
_output_shapes
:€€€€€€€€€
"
_user_specified_name
loudness:P	L
#
_output_shapes
:€€€€€€€€€
%
_user_specified_namespeechiness:J
F
#
_output_shapes
:€€€€€€€€€

_user_specified_nametempo:LH
#
_output_shapes
:€€€€€€€€€
!
_user_specified_name	valence"Ж
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ґ
serving_defaultҐ
-
ID'
serving_default_ID:0€€€€€€€€€
A
acousticness1
serving_default_acousticness:0€€€€€€€€€
A
danceability1
serving_default_danceability:0€€€€€€€€€
9
duration-
serving_default_duration:0	€€€€€€€€€
5
energy+
serving_default_energy:0€€€€€€€€€
I
instrumentalness5
"serving_default_instrumentalness:0€€€€€€€€€
/
key(
serving_default_key:0	€€€€€€€€€
9
liveness-
serving_default_liveness:0€€€€€€€€€
9
loudness-
serving_default_loudness:0€€€€€€€€€
?
speechiness0
serving_default_speechiness:0€€€€€€€€€
3
tempo*
serving_default_tempo:0€€€€€€€€€
7
valence,
serving_default_valence:0€€€€€€€€€<
output_10
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:№Џ
–
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
_build_input_shape

signatures"
_tf_keras_sequential
Ћ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_feature_columns

_resources"
_tf_keras_layer
ї
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias"
_tf_keras_layer
ї
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses

+kernel
,bias"
_tf_keras_layer
ї
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias"
_tf_keras_layer
ї
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

;kernel
<bias"
_tf_keras_layer
ї
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses

Ckernel
Dbias"
_tf_keras_layer
ї
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses

Kkernel
Lbias"
_tf_keras_layer
Љ
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses
S_random_generator"
_tf_keras_layer
ї
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses

Zkernel
[bias"
_tf_keras_layer
ї
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses

bkernel
cbias"
_tf_keras_layer
Ц
#0
$1
+2
,3
34
45
;6
<7
C8
D9
K10
L11
Z12
[13
b14
c15"
trackable_list_wrapper
Ц
#0
$1
+2
,3
34
45
;6
<7
C8
D9
K10
L11
Z12
[13
b14
c15"
trackable_list_wrapper
 "
trackable_list_wrapper
 
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
б
itrace_0
jtrace_1
ktrace_2
ltrace_32ц
+__inference_sequential_layer_call_fn_988376
+__inference_sequential_layer_call_fn_989083
+__inference_sequential_layer_call_fn_989131
+__inference_sequential_layer_call_fn_988851њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zitrace_0zjtrace_1zktrace_2zltrace_3
Ќ
mtrace_0
ntrace_1
otrace_2
ptrace_32в
F__inference_sequential_layer_call_and_return_conditional_losses_989326
F__inference_sequential_layer_call_and_return_conditional_losses_989528
F__inference_sequential_layer_call_and_return_conditional_losses_988917
F__inference_sequential_layer_call_and_return_conditional_losses_988983њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zmtrace_0zntrace_1zotrace_2zptrace_3
љBЇ
!__inference__wrapped_model_988028IDacousticnessdanceabilitydurationenergyinstrumentalnesskeylivenessloudnessspeechinesstempovalence"Ш
С≤Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ь
q
_variables
r_iterations
s_learning_rate
t_index_dict
u
_momentums
v_velocities
w_update_step_xla"
experimentalOptimizer
 "
trackable_dict_wrapper
,
xserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≠
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
п
~trace_0
trace_12Є
/__inference_dense_features_layer_call_fn_989544
/__inference_dense_features_layer_call_fn_989560”
 ≤∆
FullArgSpecE
args=Ъ:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z~trace_0ztrace_1
©
Аtrace_0
Бtrace_12о
J__inference_dense_features_layer_call_and_return_conditional_losses_989689
J__inference_dense_features_layer_call_and_return_conditional_losses_989818”
 ≤∆
FullArgSpecE
args=Ъ:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zАtrace_0zБtrace_1
 "
trackable_list_wrapper
"
_generic_user_object
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
м
Зtrace_02Ќ
&__inference_dense_layer_call_fn_989827Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЗtrace_0
З
Иtrace_02и
A__inference_dense_layer_call_and_return_conditional_losses_989838Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zИtrace_0
):'@2sequential/dense/kernel
#:!@2sequential/dense/bias
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Йnon_trainable_variables
Кlayers
Лmetrics
 Мlayer_regularization_losses
Нlayer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
о
Оtrace_02ѕ
(__inference_dense_1_layer_call_fn_989847Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zОtrace_0
Й
Пtrace_02к
C__inference_dense_1_layer_call_and_return_conditional_losses_989858Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zПtrace_0
+:)@@2sequential/dense_1/kernel
%:#@2sequential/dense_1/bias
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Рnon_trainable_variables
Сlayers
Тmetrics
 Уlayer_regularization_losses
Фlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
о
Хtrace_02ѕ
(__inference_dense_2_layer_call_fn_989867Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zХtrace_0
Й
Цtrace_02к
C__inference_dense_2_layer_call_and_return_conditional_losses_989878Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЦtrace_0
+:)@@2sequential/dense_2/kernel
%:#@2sequential/dense_2/bias
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Чnon_trainable_variables
Шlayers
Щmetrics
 Ъlayer_regularization_losses
Ыlayer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
о
Ьtrace_02ѕ
(__inference_dense_3_layer_call_fn_989887Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЬtrace_0
Й
Эtrace_02к
C__inference_dense_3_layer_call_and_return_conditional_losses_989898Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЭtrace_0
,:*	@А2sequential/dense_3/kernel
&:$А2sequential/dense_3/bias
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Юnon_trainable_variables
Яlayers
†metrics
 °layer_regularization_losses
Ґlayer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
о
£trace_02ѕ
(__inference_dense_4_layer_call_fn_989907Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z£trace_0
Й
§trace_02к
C__inference_dense_4_layer_call_and_return_conditional_losses_989918Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z§trace_0
-:+
АА2sequential/dense_4/kernel
&:$А2sequential/dense_4/bias
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
•non_trainable_variables
¶layers
Іmetrics
 ®layer_regularization_losses
©layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
о
™trace_02ѕ
(__inference_dense_5_layer_call_fn_989927Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z™trace_0
Й
Ђtrace_02к
C__inference_dense_5_layer_call_and_return_conditional_losses_989938Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЂtrace_0
-:+
АА2sequential/dense_5/kernel
&:$А2sequential/dense_5/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
≤
ђnon_trainable_variables
≠layers
Ѓmetrics
 ѓlayer_regularization_losses
∞layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
≈
±trace_0
≤trace_12К
(__inference_dropout_layer_call_fn_989943
(__inference_dropout_layer_call_fn_989948≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z±trace_0z≤trace_1
ы
≥trace_0
іtrace_12ј
C__inference_dropout_layer_call_and_return_conditional_losses_989953
C__inference_dropout_layer_call_and_return_conditional_losses_989965≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z≥trace_0zіtrace_1
"
_generic_user_object
.
Z0
[1"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
µnon_trainable_variables
ґlayers
Јmetrics
 Єlayer_regularization_losses
єlayer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
о
Їtrace_02ѕ
(__inference_dense_6_layer_call_fn_989974Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЇtrace_0
Й
їtrace_02к
C__inference_dense_6_layer_call_and_return_conditional_losses_989985Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zїtrace_0
-:+
АА2sequential/dense_6/kernel
&:$А2sequential/dense_6/bias
.
b0
c1"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
≤
Љnon_trainable_variables
љlayers
Њmetrics
 њlayer_regularization_losses
јlayer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
о
Ѕtrace_02ѕ
(__inference_dense_7_layer_call_fn_989994Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 zЅtrace_0
Й
¬trace_02к
C__inference_dense_7_layer_call_and_return_conditional_losses_990005Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 z¬trace_0
,:*	А2sequential/dense_7/kernel
%:#2sequential/dense_7/bias
 "
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
trackable_list_wrapper
0
√0
ƒ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
оBл
+__inference_sequential_layer_call_fn_988376IDacousticnessdanceabilitydurationenergyinstrumentalnesskeylivenessloudnessspeechinesstempovalence"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
¬Bњ
+__inference_sequential_layer_call_fn_989083	inputs_idinputs_acousticnessinputs_danceabilityinputs_durationinputs_energyinputs_instrumentalness
inputs_keyinputs_livenessinputs_loudnessinputs_speechinessinputs_tempoinputs_valence"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
¬Bњ
+__inference_sequential_layer_call_fn_989131	inputs_idinputs_acousticnessinputs_danceabilityinputs_durationinputs_energyinputs_instrumentalness
inputs_keyinputs_livenessinputs_loudnessinputs_speechinessinputs_tempoinputs_valence"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
оBл
+__inference_sequential_layer_call_fn_988851IDacousticnessdanceabilitydurationenergyinstrumentalnesskeylivenessloudnessspeechinesstempovalence"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЁBЏ
F__inference_sequential_layer_call_and_return_conditional_losses_989326	inputs_idinputs_acousticnessinputs_danceabilityinputs_durationinputs_energyinputs_instrumentalness
inputs_keyinputs_livenessinputs_loudnessinputs_speechinessinputs_tempoinputs_valence"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЁBЏ
F__inference_sequential_layer_call_and_return_conditional_losses_989528	inputs_idinputs_acousticnessinputs_danceabilityinputs_durationinputs_energyinputs_instrumentalness
inputs_keyinputs_livenessinputs_loudnessinputs_speechinessinputs_tempoinputs_valence"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЙBЖ
F__inference_sequential_layer_call_and_return_conditional_losses_988917IDacousticnessdanceabilitydurationenergyinstrumentalnesskeylivenessloudnessspeechinesstempovalence"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ЙBЖ
F__inference_sequential_layer_call_and_return_conditional_losses_988983IDacousticnessdanceabilitydurationenergyinstrumentalnesskeylivenessloudnessspeechinesstempovalence"њ
ґ≤≤
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Њ
r0
≈1
∆2
«3
»4
…5
 6
Ћ7
ћ8
Ќ9
ќ10
ѕ11
–12
—13
“14
”15
‘16
’17
÷18
„19
Ў20
ў21
Џ22
џ23
№24
Ё25
ё26
я27
а28
б29
в30
г31
д32"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
¶
≈0
«1
…2
Ћ3
Ќ4
ѕ5
—6
”7
’8
„9
ў10
џ11
Ё12
я13
б14
г15"
trackable_list_wrapper
¶
∆0
»1
 2
ћ3
ќ4
–5
“6
‘7
÷8
Ў9
Џ10
№11
ё12
а13
в14
д15"
trackable_list_wrapper
њ2Љє
Ѓ≤™
FullArgSpec2
args*Ъ'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 0
ЇBЈ
$__inference_signature_wrapper_989035IDacousticnessdanceabilitydurationenergyinstrumentalnesskeylivenessloudnessspeechinesstempovalence"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
тBп
/__inference_dense_features_layer_call_fn_989544features_idfeatures_acousticnessfeatures_danceabilityfeatures_durationfeatures_energyfeatures_instrumentalnessfeatures_keyfeatures_livenessfeatures_loudnessfeatures_speechinessfeatures_tempofeatures_valence"”
 ≤∆
FullArgSpecE
args=Ъ:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
тBп
/__inference_dense_features_layer_call_fn_989560features_idfeatures_acousticnessfeatures_danceabilityfeatures_durationfeatures_energyfeatures_instrumentalnessfeatures_keyfeatures_livenessfeatures_loudnessfeatures_speechinessfeatures_tempofeatures_valence"”
 ≤∆
FullArgSpecE
args=Ъ:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
НBК
J__inference_dense_features_layer_call_and_return_conditional_losses_989689features_idfeatures_acousticnessfeatures_danceabilityfeatures_durationfeatures_energyfeatures_instrumentalnessfeatures_keyfeatures_livenessfeatures_loudnessfeatures_speechinessfeatures_tempofeatures_valence"”
 ≤∆
FullArgSpecE
args=Ъ:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
НBК
J__inference_dense_features_layer_call_and_return_conditional_losses_989818features_idfeatures_acousticnessfeatures_danceabilityfeatures_durationfeatures_energyfeatures_instrumentalnessfeatures_keyfeatures_livenessfeatures_loudnessfeatures_speechinessfeatures_tempofeatures_valence"”
 ≤∆
FullArgSpecE
args=Ъ:
jself

jfeatures
jcols_to_output_tensors

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЏB„
&__inference_dense_layer_call_fn_989827inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
хBт
A__inference_dense_layer_call_and_return_conditional_losses_989838inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№Bў
(__inference_dense_1_layer_call_fn_989847inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
чBф
C__inference_dense_1_layer_call_and_return_conditional_losses_989858inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№Bў
(__inference_dense_2_layer_call_fn_989867inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
чBф
C__inference_dense_2_layer_call_and_return_conditional_losses_989878inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№Bў
(__inference_dense_3_layer_call_fn_989887inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
чBф
C__inference_dense_3_layer_call_and_return_conditional_losses_989898inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№Bў
(__inference_dense_4_layer_call_fn_989907inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
чBф
C__inference_dense_4_layer_call_and_return_conditional_losses_989918inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№Bў
(__inference_dense_5_layer_call_fn_989927inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
чBф
C__inference_dense_5_layer_call_and_return_conditional_losses_989938inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
нBк
(__inference_dropout_layer_call_fn_989943inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
нBк
(__inference_dropout_layer_call_fn_989948inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ИBЕ
C__inference_dropout_layer_call_and_return_conditional_losses_989953inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ИBЕ
C__inference_dropout_layer_call_and_return_conditional_losses_989965inputs"≥
™≤¶
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№Bў
(__inference_dense_6_layer_call_fn_989974inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
чBф
C__inference_dense_6_layer_call_and_return_conditional_losses_989985inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
№Bў
(__inference_dense_7_layer_call_fn_989994inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
чBф
C__inference_dense_7_layer_call_and_return_conditional_losses_990005inputs"Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
R
е	variables
ж	keras_api

зtotal

иcount"
_tf_keras_metric
c
й	variables
к	keras_api

лtotal

мcount
н
_fn_kwargs"
_tf_keras_metric
.:,@2Adam/m/sequential/dense/kernel
.:,@2Adam/v/sequential/dense/kernel
(:&@2Adam/m/sequential/dense/bias
(:&@2Adam/v/sequential/dense/bias
0:.@@2 Adam/m/sequential/dense_1/kernel
0:.@@2 Adam/v/sequential/dense_1/kernel
*:(@2Adam/m/sequential/dense_1/bias
*:(@2Adam/v/sequential/dense_1/bias
0:.@@2 Adam/m/sequential/dense_2/kernel
0:.@@2 Adam/v/sequential/dense_2/kernel
*:(@2Adam/m/sequential/dense_2/bias
*:(@2Adam/v/sequential/dense_2/bias
1:/	@А2 Adam/m/sequential/dense_3/kernel
1:/	@А2 Adam/v/sequential/dense_3/kernel
+:)А2Adam/m/sequential/dense_3/bias
+:)А2Adam/v/sequential/dense_3/bias
2:0
АА2 Adam/m/sequential/dense_4/kernel
2:0
АА2 Adam/v/sequential/dense_4/kernel
+:)А2Adam/m/sequential/dense_4/bias
+:)А2Adam/v/sequential/dense_4/bias
2:0
АА2 Adam/m/sequential/dense_5/kernel
2:0
АА2 Adam/v/sequential/dense_5/kernel
+:)А2Adam/m/sequential/dense_5/bias
+:)А2Adam/v/sequential/dense_5/bias
2:0
АА2 Adam/m/sequential/dense_6/kernel
2:0
АА2 Adam/v/sequential/dense_6/kernel
+:)А2Adam/m/sequential/dense_6/bias
+:)А2Adam/v/sequential/dense_6/bias
1:/	А2 Adam/m/sequential/dense_7/kernel
1:/	А2 Adam/v/sequential/dense_7/kernel
*:(2Adam/m/sequential/dense_7/bias
*:(2Adam/v/sequential/dense_7/bias
0
з0
и1"
trackable_list_wrapper
.
е	variables"
_generic_user_object
:  (2total
:  (2count
0
л0
м1"
trackable_list_wrapper
.
й	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapperЫ
!__inference__wrapped_model_988028х#$+,34;<CDKLZ[bcЂҐІ
ЯҐЫ
Ш™Ф

IDК
ID€€€€€€€€€
2
acousticness"К
acousticness€€€€€€€€€
2
danceability"К
danceability€€€€€€€€€
*
durationК
duration€€€€€€€€€	
&
energyК
energy€€€€€€€€€
:
instrumentalness&К#
instrumentalness€€€€€€€€€
 
keyК
key€€€€€€€€€	
*
livenessК
liveness€€€€€€€€€
*
loudnessК
loudness€€€€€€€€€
0
speechiness!К
speechiness€€€€€€€€€
$
tempoК
tempo€€€€€€€€€
(
valenceК
valence€€€€€€€€€
™ "3™0
.
output_1"К
output_1€€€€€€€€€™
C__inference_dense_1_layer_call_and_return_conditional_losses_989858c+,/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ ",Ґ)
"К
tensor_0€€€€€€€€€@
Ъ Д
(__inference_dense_1_layer_call_fn_989847X+,/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "!К
unknown€€€€€€€€€@™
C__inference_dense_2_layer_call_and_return_conditional_losses_989878c34/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ ",Ґ)
"К
tensor_0€€€€€€€€€@
Ъ Д
(__inference_dense_2_layer_call_fn_989867X34/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "!К
unknown€€€€€€€€€@Ђ
C__inference_dense_3_layer_call_and_return_conditional_losses_989898d;</Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "-Ґ*
#К 
tensor_0€€€€€€€€€А
Ъ Е
(__inference_dense_3_layer_call_fn_989887Y;</Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ ""К
unknown€€€€€€€€€Ађ
C__inference_dense_4_layer_call_and_return_conditional_losses_989918eCD0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "-Ґ*
#К 
tensor_0€€€€€€€€€А
Ъ Ж
(__inference_dense_4_layer_call_fn_989907ZCD0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ ""К
unknown€€€€€€€€€Ађ
C__inference_dense_5_layer_call_and_return_conditional_losses_989938eKL0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "-Ґ*
#К 
tensor_0€€€€€€€€€А
Ъ Ж
(__inference_dense_5_layer_call_fn_989927ZKL0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ ""К
unknown€€€€€€€€€Ађ
C__inference_dense_6_layer_call_and_return_conditional_losses_989985eZ[0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "-Ґ*
#К 
tensor_0€€€€€€€€€А
Ъ Ж
(__inference_dense_6_layer_call_fn_989974ZZ[0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ ""К
unknown€€€€€€€€€АЂ
C__inference_dense_7_layer_call_and_return_conditional_losses_990005dbc0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ Е
(__inference_dense_7_layer_call_fn_989994Ybc0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "!К
unknown€€€€€€€€€Я
J__inference_dense_features_layer_call_and_return_conditional_losses_989689–ЯҐЫ
УҐП
Д™А
'
ID!К
features_id€€€€€€€€€
;
acousticness+К(
features_acousticness€€€€€€€€€
;
danceability+К(
features_danceability€€€€€€€€€
3
duration'К$
features_duration€€€€€€€€€	
/
energy%К"
features_energy€€€€€€€€€
C
instrumentalness/К,
features_instrumentalness€€€€€€€€€
)
key"К
features_key€€€€€€€€€	
3
liveness'К$
features_liveness€€€€€€€€€
3
loudness'К$
features_loudness€€€€€€€€€
9
speechiness*К'
features_speechiness€€€€€€€€€
-
tempo$К!
features_tempo€€€€€€€€€
1
valence&К#
features_valence€€€€€€€€€

 
p 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ Я
J__inference_dense_features_layer_call_and_return_conditional_losses_989818–ЯҐЫ
УҐП
Д™А
'
ID!К
features_id€€€€€€€€€
;
acousticness+К(
features_acousticness€€€€€€€€€
;
danceability+К(
features_danceability€€€€€€€€€
3
duration'К$
features_duration€€€€€€€€€	
/
energy%К"
features_energy€€€€€€€€€
C
instrumentalness/К,
features_instrumentalness€€€€€€€€€
)
key"К
features_key€€€€€€€€€	
3
liveness'К$
features_liveness€€€€€€€€€
3
loudness'К$
features_loudness€€€€€€€€€
9
speechiness*К'
features_speechiness€€€€€€€€€
-
tempo$К!
features_tempo€€€€€€€€€
1
valence&К#
features_valence€€€€€€€€€

 
p
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ щ
/__inference_dense_features_layer_call_fn_989544≈ЯҐЫ
УҐП
Д™А
'
ID!К
features_id€€€€€€€€€
;
acousticness+К(
features_acousticness€€€€€€€€€
;
danceability+К(
features_danceability€€€€€€€€€
3
duration'К$
features_duration€€€€€€€€€	
/
energy%К"
features_energy€€€€€€€€€
C
instrumentalness/К,
features_instrumentalness€€€€€€€€€
)
key"К
features_key€€€€€€€€€	
3
liveness'К$
features_liveness€€€€€€€€€
3
loudness'К$
features_loudness€€€€€€€€€
9
speechiness*К'
features_speechiness€€€€€€€€€
-
tempo$К!
features_tempo€€€€€€€€€
1
valence&К#
features_valence€€€€€€€€€

 
p 
™ "!К
unknown€€€€€€€€€щ
/__inference_dense_features_layer_call_fn_989560≈ЯҐЫ
УҐП
Д™А
'
ID!К
features_id€€€€€€€€€
;
acousticness+К(
features_acousticness€€€€€€€€€
;
danceability+К(
features_danceability€€€€€€€€€
3
duration'К$
features_duration€€€€€€€€€	
/
energy%К"
features_energy€€€€€€€€€
C
instrumentalness/К,
features_instrumentalness€€€€€€€€€
)
key"К
features_key€€€€€€€€€	
3
liveness'К$
features_liveness€€€€€€€€€
3
loudness'К$
features_loudness€€€€€€€€€
9
speechiness*К'
features_speechiness€€€€€€€€€
-
tempo$К!
features_tempo€€€€€€€€€
1
valence&К#
features_valence€€€€€€€€€

 
p
™ "!К
unknown€€€€€€€€€®
A__inference_dense_layer_call_and_return_conditional_losses_989838c#$/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ ",Ґ)
"К
tensor_0€€€€€€€€€@
Ъ В
&__inference_dense_layer_call_fn_989827X#$/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "!К
unknown€€€€€€€€€@ђ
C__inference_dropout_layer_call_and_return_conditional_losses_989953e4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ "-Ґ*
#К 
tensor_0€€€€€€€€€А
Ъ ђ
C__inference_dropout_layer_call_and_return_conditional_losses_989965e4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ "-Ґ*
#К 
tensor_0€€€€€€€€€А
Ъ Ж
(__inference_dropout_layer_call_fn_989943Z4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p 
™ ""К
unknown€€€€€€€€€АЖ
(__inference_dropout_layer_call_fn_989948Z4Ґ1
*Ґ'
!К
inputs€€€€€€€€€А
p
™ ""К
unknown€€€€€€€€€АЅ
F__inference_sequential_layer_call_and_return_conditional_losses_988917ц#$+,34;<CDKLZ[bc≥Ґѓ
ІҐ£
Ш™Ф

IDК
ID€€€€€€€€€
2
acousticness"К
acousticness€€€€€€€€€
2
danceability"К
danceability€€€€€€€€€
*
durationК
duration€€€€€€€€€	
&
energyК
energy€€€€€€€€€
:
instrumentalness&К#
instrumentalness€€€€€€€€€
 
keyК
key€€€€€€€€€	
*
livenessК
liveness€€€€€€€€€
*
loudnessК
loudness€€€€€€€€€
0
speechiness!К
speechiness€€€€€€€€€
$
tempoК
tempo€€€€€€€€€
(
valenceК
valence€€€€€€€€€
p 

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ Ѕ
F__inference_sequential_layer_call_and_return_conditional_losses_988983ц#$+,34;<CDKLZ[bc≥Ґѓ
ІҐ£
Ш™Ф

IDК
ID€€€€€€€€€
2
acousticness"К
acousticness€€€€€€€€€
2
danceability"К
danceability€€€€€€€€€
*
durationК
duration€€€€€€€€€	
&
energyК
energy€€€€€€€€€
:
instrumentalness&К#
instrumentalness€€€€€€€€€
 
keyК
key€€€€€€€€€	
*
livenessК
liveness€€€€€€€€€
*
loudnessК
loudness€€€€€€€€€
0
speechiness!К
speechiness€€€€€€€€€
$
tempoК
tempo€€€€€€€€€
(
valenceК
valence€€€€€€€€€
p

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ Х
F__inference_sequential_layer_call_and_return_conditional_losses_989326 #$+,34;<CDKLZ[bcЗҐГ
ыҐч
м™и
%
IDК
	inputs_id€€€€€€€€€
9
acousticness)К&
inputs_acousticness€€€€€€€€€
9
danceability)К&
inputs_danceability€€€€€€€€€
1
duration%К"
inputs_duration€€€€€€€€€	
-
energy#К 
inputs_energy€€€€€€€€€
A
instrumentalness-К*
inputs_instrumentalness€€€€€€€€€
'
key К

inputs_key€€€€€€€€€	
1
liveness%К"
inputs_liveness€€€€€€€€€
1
loudness%К"
inputs_loudness€€€€€€€€€
7
speechiness(К%
inputs_speechiness€€€€€€€€€
+
tempo"К
inputs_tempo€€€€€€€€€
/
valence$К!
inputs_valence€€€€€€€€€
p 

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ Х
F__inference_sequential_layer_call_and_return_conditional_losses_989528 #$+,34;<CDKLZ[bcЗҐГ
ыҐч
м™и
%
IDК
	inputs_id€€€€€€€€€
9
acousticness)К&
inputs_acousticness€€€€€€€€€
9
danceability)К&
inputs_danceability€€€€€€€€€
1
duration%К"
inputs_duration€€€€€€€€€	
-
energy#К 
inputs_energy€€€€€€€€€
A
instrumentalness-К*
inputs_instrumentalness€€€€€€€€€
'
key К

inputs_key€€€€€€€€€	
1
liveness%К"
inputs_liveness€€€€€€€€€
1
loudness%К"
inputs_loudness€€€€€€€€€
7
speechiness(К%
inputs_speechiness€€€€€€€€€
+
tempo"К
inputs_tempo€€€€€€€€€
/
valence$К!
inputs_valence€€€€€€€€€
p

 
™ ",Ґ)
"К
tensor_0€€€€€€€€€
Ъ Ы
+__inference_sequential_layer_call_fn_988376л#$+,34;<CDKLZ[bc≥Ґѓ
ІҐ£
Ш™Ф

IDК
ID€€€€€€€€€
2
acousticness"К
acousticness€€€€€€€€€
2
danceability"К
danceability€€€€€€€€€
*
durationК
duration€€€€€€€€€	
&
energyК
energy€€€€€€€€€
:
instrumentalness&К#
instrumentalness€€€€€€€€€
 
keyК
key€€€€€€€€€	
*
livenessК
liveness€€€€€€€€€
*
loudnessК
loudness€€€€€€€€€
0
speechiness!К
speechiness€€€€€€€€€
$
tempoК
tempo€€€€€€€€€
(
valenceК
valence€€€€€€€€€
p 

 
™ "!К
unknown€€€€€€€€€Ы
+__inference_sequential_layer_call_fn_988851л#$+,34;<CDKLZ[bc≥Ґѓ
ІҐ£
Ш™Ф

IDК
ID€€€€€€€€€
2
acousticness"К
acousticness€€€€€€€€€
2
danceability"К
danceability€€€€€€€€€
*
durationК
duration€€€€€€€€€	
&
energyК
energy€€€€€€€€€
:
instrumentalness&К#
instrumentalness€€€€€€€€€
 
keyК
key€€€€€€€€€	
*
livenessК
liveness€€€€€€€€€
*
loudnessК
loudness€€€€€€€€€
0
speechiness!К
speechiness€€€€€€€€€
$
tempoК
tempo€€€€€€€€€
(
valenceК
valence€€€€€€€€€
p

 
™ "!К
unknown€€€€€€€€€п
+__inference_sequential_layer_call_fn_989083њ#$+,34;<CDKLZ[bcЗҐГ
ыҐч
м™и
%
IDК
	inputs_id€€€€€€€€€
9
acousticness)К&
inputs_acousticness€€€€€€€€€
9
danceability)К&
inputs_danceability€€€€€€€€€
1
duration%К"
inputs_duration€€€€€€€€€	
-
energy#К 
inputs_energy€€€€€€€€€
A
instrumentalness-К*
inputs_instrumentalness€€€€€€€€€
'
key К

inputs_key€€€€€€€€€	
1
liveness%К"
inputs_liveness€€€€€€€€€
1
loudness%К"
inputs_loudness€€€€€€€€€
7
speechiness(К%
inputs_speechiness€€€€€€€€€
+
tempo"К
inputs_tempo€€€€€€€€€
/
valence$К!
inputs_valence€€€€€€€€€
p 

 
™ "!К
unknown€€€€€€€€€п
+__inference_sequential_layer_call_fn_989131њ#$+,34;<CDKLZ[bcЗҐГ
ыҐч
м™и
%
IDК
	inputs_id€€€€€€€€€
9
acousticness)К&
inputs_acousticness€€€€€€€€€
9
danceability)К&
inputs_danceability€€€€€€€€€
1
duration%К"
inputs_duration€€€€€€€€€	
-
energy#К 
inputs_energy€€€€€€€€€
A
instrumentalness-К*
inputs_instrumentalness€€€€€€€€€
'
key К

inputs_key€€€€€€€€€	
1
liveness%К"
inputs_liveness€€€€€€€€€
1
loudness%К"
inputs_loudness€€€€€€€€€
7
speechiness(К%
inputs_speechiness€€€€€€€€€
+
tempo"К
inputs_tempo€€€€€€€€€
/
valence$К!
inputs_valence€€€€€€€€€
p

 
™ "!К
unknown€€€€€€€€€Ч
$__inference_signature_wrapper_989035о#$+,34;<CDKLZ[bc§Ґ†
Ґ 
Ш™Ф

IDК
id€€€€€€€€€
2
acousticness"К
acousticness€€€€€€€€€
2
danceability"К
danceability€€€€€€€€€
*
durationК
duration€€€€€€€€€	
&
energyК
energy€€€€€€€€€
:
instrumentalness&К#
instrumentalness€€€€€€€€€
 
keyК
key€€€€€€€€€	
*
livenessК
liveness€€€€€€€€€
*
loudnessК
loudness€€€€€€€€€
0
speechiness!К
speechiness€€€€€€€€€
$
tempoК
tempo€€€€€€€€€
(
valenceК
valence€€€€€€€€€"3™0
.
output_1"К
output_1€€€€€€€€€