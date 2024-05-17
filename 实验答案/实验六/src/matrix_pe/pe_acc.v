module pe_acc(
  input  [1023:0] mult_result,
  output [  31:0] acc_result
);

genvar i;
genvar j;

// *******************************************************************
// int16 add tree
// *******************************************************************
wire [31:0] int16_result[5:0][31:0];
for(i=0; i<=5; i=i+1)
begin:int16_add_tree
  for(j=0; j<(32/(2**i)); j=j+1)
  begin:int16_adder
    if(i==0) begin
      assign int16_result[0][j] = mult_result[31+32*j:0+32*j];
    end else begin
      assign int16_result[i][j] = int16_result[i-1][j*2] + int16_result[i-1][j*2 + 1];
    end
  end
end

assign acc_result = int16_result[5][0];

endmodule
