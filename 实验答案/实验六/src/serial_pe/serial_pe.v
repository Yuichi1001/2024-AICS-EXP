module serial_pe(
  input                clk,
  input                rst_n,
  input  signed [15:0] neuron,  //INT16神经元分量
  input  signed [15:0] weight,  //INT16权值分量
  input         [ 1:0] ctl,     //输入控制 ctl[0]表示第一个输入元素 ctl[1]表示最后一个
  input                vld_i,   //输入和控制信号有效
  output        [31:0] result,  
  output reg           vld_o    //输出有效
);

/* 乘法器 */ /* TODO*/
wire signed [31:0] mult_res = neuron * weight;
reg [31:0] psum_r;

/* 加法器 */ /* TODO*/
wire [31:0] psum_d = ctl[0] ? mult_res : mult_res + psum_r;
//wire [31:0] psum_d =mult_res + psum_r;

/* 部分和寄存器 */
always@(posedge clk or negedge rst_n)
if(!rst_n) begin
  psum_r <= 32'h0;
end else if(vld_i) begin
  psum_r <= psum_d;
end

always@(posedge clk or negedge rst_n)
if(!rst_n) begin
  vld_o <= 1'b0;
end else if(ctl[1]) begin
  vld_o <= 1'b1;
end else begin
  vld_o <= 1'b0;
end

assign result = psum_r;

endmodule
